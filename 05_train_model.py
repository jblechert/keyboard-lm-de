#!/usr/bin/env python3
"""
Trains a Llama-based German keyboard language model from scratch.

Architecture (~36M Parameter, entspricht FUTO English model):
  8 Layer × 512 Hidden Dims × 8 Attention Heads × 1024 FFN

Input:
  data/tatoeba_de.txt              Tatoeba German sentences
  data/tokenizer/de_keyboard.model SentencePiece Tokenizer

Output:
  data/model_hf/              HuggingFace Checkpoint

Usage:
  .venv_ml/bin/python 05_train_model.py [--steps 100000] [--resume]
"""

import argparse
import math
import os
import random
from pathlib import Path

import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import IterableDataset

# ── Pfade ─────────────────────────────────────────────────────────────────────
SP_MODEL       = Path("data/tokenizer/de_keyboard.model")
TATOEBA_TXT    = Path("data/tatoeba_de.txt")
C4_TXT         = Path("data/c4_de.txt")
SYNTHETIC_TXT  = Path("data/synthetic_de.txt")
OUTPUT_DIR     = Path("data/model_hf")

# Sampling-Gewichte (relativ zueinander)
# Tatoeba: sauber, alltagsnah            → 3×
# mC4:     groß, gemischt                → 1×
# Synthese: keyboard-spezifisch, klein   → 6×
TATOEBA_WEIGHT   = 3
C4_WEIGHT        = 1
SYNTHETIC_WEIGHT = 6

# ── Modell-Architektur (FUTO-kompatibel) ──────────────────────────────────────
MODEL_CONFIG = dict(
    hidden_size=512,
    num_hidden_layers=8,
    num_attention_heads=8,
    intermediate_size=1024,
    max_position_embeddings=256,   # Keyboard-LM braucht keinen langen Kontext
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
    attention_bias=False,
    hidden_act="silu",
)

# ── Training-Hyperparameter ───────────────────────────────────────────────────
CONTEXT_LEN        = 256
BATCH_SIZE         = 32    # pro GPU
GRAD_ACCUM         = 8     # effektive Batchgröße = 256
LR                 = 3e-4
LR_WARMUP_STEPS    = 1000
WEIGHT_DECAY       = 0.1
MAX_GRAD_NORM      = 1.0
SAVE_STEPS         = 5_000
LOGGING_STEPS      = 200
SNAPSHOT_DIR       = Path("data/snapshots")

# ── Tokenizer ─────────────────────────────────────────────────────────────────

def load_tokenizer() -> LlamaTokenizer:
    tok = LlamaTokenizer(vocab_file=str(SP_MODEL), legacy=False)
    tok.add_special_tokens({
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<unk>",
    })
    # FUTO-Autocorrect-Tokens sind bereits im SP-Modell als user_defined_symbols
    return tok


# ── Dataset ───────────────────────────────────────────────────────────────────

def line_generator(path: Path):
    """Endloser Generator über eine Textdatei (wiederholt sich nach Ende)."""
    while True:
        with path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        random.shuffle(lines)
        yield from lines


def mixed_generator():
    """Mischt alle verfügbaren Quellen gewichtet."""
    sources = []
    if TATOEBA_TXT.exists():
        sources.append((line_generator(TATOEBA_TXT), TATOEBA_WEIGHT))
    if C4_TXT.exists():
        sources.append((line_generator(C4_TXT), C4_WEIGHT))
    if SYNTHETIC_TXT.exists():
        sources.append((line_generator(SYNTHETIC_TXT), SYNTHETIC_WEIGHT))

    if not sources:
        raise FileNotFoundError("Keine Trainingsdaten gefunden.")

    gens, weights = zip(*sources)
    total = sum(weights)
    thresholds = []
    acc = 0
    for w in weights:
        acc += w / total
        thresholds.append(acc)

    while True:
        r = random.random()
        for gen, threshold in zip(gens, thresholds):
            if r < threshold:
                yield next(gen).strip()
                break


def tokenize_and_chunk(tokenizer: LlamaTokenizer, context_len: int):
    """
    Gibt einen IterableDataset zurück der tokenisierte Chunks der Länge
    context_len liefert. Sätze werden zu langen Sequenzen verkettet
    (mit <s>/</ s> als Trennzeichen) und dann in Chunks geschnitten.
    """
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id

    def gen():
        buf = []
        for line in mixed_generator():
            if not line:
                continue
            ids = tokenizer.encode(line, add_special_tokens=False)
            buf += [bos] + ids + [eos]
            while len(buf) >= context_len:
                chunk = buf[:context_len]
                buf   = buf[context_len:]
                yield {"input_ids": chunk}

    return IterableDataset.from_generator(gen)


# ── Milestone-Callback ────────────────────────────────────────────────────────

class MilestoneCallback(TrainerCallback):
    """Speichert permanente Snapshots bei bestimmten Schritten."""

    def __init__(self, milestones: set[int], snapshot_dir: Path, tokenizer):
        self.milestones   = milestones
        self.snapshot_dir = snapshot_dir
        self.tokenizer    = tokenizer

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step in self.milestones:
            out = self.snapshot_dir / f"step_{state.global_step:06d}"
            out.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(out))
            self.tokenizer.save_pretrained(str(out))
            print(f"\n[Milestone] → {out}", flush=True)


# ── Modell ────────────────────────────────────────────────────────────────────

def build_model(tokenizer: LlamaTokenizer) -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        **MODEL_CONFIG,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = LlamaForCausalLM(config)
    params = sum(p.numel() for p in model.parameters())
    print(f"Modell: {params/1e6:.1f}M Parameter")
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",  type=int, default=200_000,
                        help="Trainingsschritte gesamt (default: 200k)")
    parser.add_argument("--milestones", default=",".join(str(s) for s in range(50_000, 200_001, 10_000)),
                        help="Kommagetrennte Schritte für permanente Snapshots "
                             "(default: 50k–200k alle 10k)")
    parser.add_argument("--resume", action="store_true",
                        help="Von letztem Checkpoint weitermachen")
    args = parser.parse_args()

    milestones = {int(s) for s in args.milestones.split(",") if s.strip()}

    if not SP_MODEL.exists():
        print(f"Fehler: Tokenizer nicht gefunden: {SP_MODEL}")
        return
    if not any(p.exists() for p in [TATOEBA_TXT, C4_TXT, SYNTHETIC_TXT]):
        print("Fehler: Keine Trainingsdaten gefunden.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Lade Tokenizer ...")
    tokenizer = load_tokenizer()

    print("Baue Modell ...")
    if args.resume and (OUTPUT_DIR / "config.json").exists():
        model = LlamaForCausalLM.from_pretrained(str(OUTPUT_DIR))
        print("  → Checkpoint geladen")
    else:
        model = build_model(tokenizer)

    print("Erstelle Dataset ...")
    dataset = tokenize_and_chunk(tokenizer, CONTEXT_LEN)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        max_steps=args.steps,

        # Batch
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,

        # Optimierer
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=LR_WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        optim="adamw_torch",

        # Gradient Checkpointing deaktiviert (seq_len=256 passt in VRAM, ~30% schneller)
        gradient_checkpointing=False,
        bf16=True,              # RX 7900 XTX unterstützt BF16

        # Logging & Speichern
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,     # max. 3 Checkpoints behalten

        # Kein Eval (kein Validation-Set)
        do_eval=False,

        # Reproduzierbarkeit
        seed=42,
        dataloader_num_workers=0,  # IterableDataset verträgt kein Multiprocessing-Pickling

        report_to="none",
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,   # Causal LM, kein Masked LM
    )

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    milestone_cb = MilestoneCallback(milestones, SNAPSHOT_DIR, tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[milestone_cb],
    )

    print(f"\nStarte Training für {args.steps:,} Schritte ...")
    print(f"  Effektive Batchgröße: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  Kontext:              {CONTEXT_LEN} Tokens")
    active = []
    if TATOEBA_TXT.exists():   active.append(f"Tatoeba({TATOEBA_WEIGHT}×)")
    if C4_TXT.exists():        active.append(f"mC4({C4_WEIGHT}×)")
    if SYNTHETIC_TXT.exists(): active.append(f"Synthese({SYNTHETIC_WEIGHT}×)")
    print(f"  Corpus:               {' + '.join(active)}")
    print(f"  Snapshots bei:        {sorted(milestones)}")

    resume_from = str(OUTPUT_DIR) if args.resume else None
    trainer.train(resume_from_checkpoint=resume_from)

    print("\nSpeichere finales Modell ...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"Fertig: {OUTPUT_DIR}")
    print("Nächster Schritt: .venv_ml/bin/python 06_convert_to_gguf.py")


if __name__ == "__main__":
    main()
