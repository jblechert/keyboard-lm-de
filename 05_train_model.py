#!/usr/bin/env python3
"""
Trains a Llama-based German keyboard language model from scratch.

Architecture (~57M Parameter):
  10 Layer × 512 Hidden Dims × 8 Attention Heads × 2048 FFN

Input:
  data/tatoeba_de.txt              Tatoeba German sentences
  data/tokenizer/de_keyboard.model SentencePiece Tokenizer

Output:
  models/de_keyboard/              HuggingFace Checkpoint

v0.5-Änderungen:
  - torch.compile() für Kernel-Fusion auf RDNA3
  - Batch 64 statt 32 (GRAD_ACCUM 4, effektiv 256 gleich)
  - Neue Quellen: CRE, Raumzeit, Forschergeist, MI, CCC Congress

Usage:
  .venv_ml/bin/python 05_train_model.py [--steps 100000] [--resume]
"""

import argparse
import math
import os
import random
from pathlib import Path

import threading
import time
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

def gpu_stats() -> str:
    """Gibt GPU-Auslastung und VRAM zurück."""
    try:
        import subprocess
        r = subprocess.run(['rocm-smi', '--showuse', '--showmemuse'],
                           capture_output=True, text=True, timeout=3)
        use  = next((l.split(':')[1].strip() for l in r.stdout.splitlines() if 'GPU use' in l), '?')
        mem  = next((l.split(':')[1].strip() for l in r.stdout.splitlines() if 'GPU Memory' in l), '?')
        return f"GPU {use} | VRAM {mem}"
    except Exception:
        return "GPU-Stats nicht verfügbar"


# ── Pfade ─────────────────────────────────────────────────────────────────────
SP_MODEL            = Path("data/tokenizer/de_keyboard.model")
TATOEBA_TXT         = Path("data/tatoeba_de.txt")
C4_TXT              = Path("data/c4_de.txt")
FINEWEB2_TXT        = Path("data/fineweb2_de.txt")
SYNTHETIC_GLOB      = "data/synthetic_*.txt"
PRIVATE_TXT         = Path("data/private_de.txt")
PARLAMENTSREVUE_TXT = Path("data/parlamentsrevue_de.txt")  # CC BY-SA 4.0
LNP_TXT             = Path("data/lnp_de.txt")              # CC BY-NC-SA 3.0 DE
MINKORREKT_TXT      = Path("data/minkorrekt_de.txt")        # CC BY-NC-SA 3.0
RAUMZEIT_TXT        = Path("data/raumzeit_de.txt")          # CC BY-NC-SA 3.0 DE
FORSCHERGEIST_TXT   = Path("data/forschergeist_de.txt")     # CC BY-NC-SA 3.0 DE
CRE_TXT             = Path("data/cre_de.txt")               # CC BY-NC-SA 3.0 DE
CCC_CONGRESS_TXT    = Path("data/ccc_congress_de.txt")      # CC BY 3.0
OUTPUT_DIR          = Path("models/de_keyboard")

# ── Sampling-Gewichte ─────────────────────────────────────────────────────────
BASE_WEIGHT           = 1   # c4 und FineWeb2 gleichwertig (großer Hintergrundcorpus)
TATOEBA_WEIGHT        = 3   # sauber, alltagsnah
C4_WEIGHT             = BASE_WEIGHT
FINEWEB2_WEIGHT       = BASE_WEIGHT
SYNTHETIC_WEIGHT      = 3   # keyboard-spezifisch
PRIVATE_WEIGHT        = 2   # echter Schreibstil
PARLAMENTSREVUE_WEIGHT = 2  # gesprochenes Hochdeutsch
LNP_WEIGHT            = 2   # gesprochenes Hochdeutsch
PODCAST_WEIGHT        = 2   # alle weiteren Podcast-Quellen

# ── Modell-Architektur (FUTO-kompatibel) ──────────────────────────────────────
MODEL_CONFIG = dict(
    hidden_size=512,
    num_hidden_layers=10,
    num_attention_heads=8,
    intermediate_size=2048,
    max_position_embeddings=256,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
    attention_bias=False,
    hidden_act="silu",
)

# ── Training-Hyperparameter ───────────────────────────────────────────────────
CONTEXT_LEN        = 256
BATCH_SIZE         = 64    # v0.5: 64 statt 32 — 7900 XTX hat genug VRAM
GRAD_ACCUM         = 4     # effektive Batchgröße bleibt 256
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
    return tok


# ── Dataset ───────────────────────────────────────────────────────────────────

def _has_content(path: Path) -> bool:
    """True, wenn die Datei existiert und nicht leer ist."""
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


def line_generator(path: Path, shuffle_buffer: int = 50_000):
    """Streaming-Generator mit Shuffle-Buffer — liest nie mehr als noetig."""
    while True:
        produced = False
        buf = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                buf.append(line)
                if len(buf) >= shuffle_buffer:
                    random.shuffle(buf)
                    produced = True
                    yield from buf
                    buf = []
        if buf:
            random.shuffle(buf)
            produced = True
            yield from buf
        if not produced:
            # Leere Datei → sonst würde diese while-True-Schleife für immer
            # drehen, ohne je etwas zu liefern (Trainings-Hang).
            return


def mixed_generator(no_synthetic: bool = False):
    sources = []
    if _has_content(TATOEBA_TXT):
        sources.append((line_generator(TATOEBA_TXT), TATOEBA_WEIGHT))
    if _has_content(C4_TXT):
        sources.append((line_generator(C4_TXT), C4_WEIGHT))
    if _has_content(FINEWEB2_TXT):
        sources.append((line_generator(FINEWEB2_TXT), FINEWEB2_WEIGHT))
    if not no_synthetic:
        for syn in sorted(Path(".").glob(SYNTHETIC_GLOB)):
            if _has_content(syn):
                sources.append((line_generator(syn), SYNTHETIC_WEIGHT))
    if _has_content(PRIVATE_TXT):
        sources.append((line_generator(PRIVATE_TXT), PRIVATE_WEIGHT))
    if _has_content(PARLAMENTSREVUE_TXT):
        sources.append((line_generator(PARLAMENTSREVUE_TXT), PARLAMENTSREVUE_WEIGHT))

    for podcast in [LNP_TXT, MINKORREKT_TXT, RAUMZEIT_TXT,
                    FORSCHERGEIST_TXT, CRE_TXT, CCC_CONGRESS_TXT]:
        if _has_content(podcast):
            sources.append((line_generator(podcast), PODCAST_WEIGHT))

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


def tokenize_and_chunk(tokenizer: LlamaTokenizer, context_len: int, no_synthetic: bool = False):
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id

    def gen():
        buf = []
        for line in mixed_generator(no_synthetic=no_synthetic):
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
    def __init__(self, milestones: set[int], snapshot_dir: Path, tokenizer):
        self.milestones   = milestones
        self.snapshot_dir = snapshot_dir
        self.tokenizer    = tokenizer
        self._first_step  = True

    def on_step_begin(self, args, state, control, **kwargs):
        if self._first_step:
            self._first_step = False
            print("\n[Step 1 - Kompilierung fertig] " + gpu_stats(), flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.global_step > 0:
            loss = logs.get('loss', '?')
            lr   = logs.get('learning_rate', '?')
            print(f"  Step {state.global_step:6d} | loss={loss:.4f} lr={lr:.2e} | {gpu_stats()}",
                  flush=True)

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
    parser.add_argument("--no-synthetic", action="store_true")
    parser.add_argument("--steps",  type=int, default=200_000)
    parser.add_argument("--milestones", default=",".join(str(s) for s in range(50_000, 200_001, 10_000)))
    parser.add_argument("--version", default="v0.5")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    milestones = {int(s) for s in args.milestones.split(",") if s.strip()}
    snapshot_dir = Path("data/snapshots") / args.version

    if not SP_MODEL.exists():
        print(f"Fehler: Tokenizer nicht gefunden: {SP_MODEL}")
        return
    syn_files = sorted(Path(".").glob(SYNTHETIC_GLOB))
    if not any(p.exists() for p in [TATOEBA_TXT, C4_TXT]) and not syn_files:
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
    dataset = tokenize_and_chunk(tokenizer, CONTEXT_LEN, no_synthetic=args.no_synthetic)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        max_steps=args.steps,

        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,

        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=LR_WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        optim="adamw_torch",

        gradient_checkpointing=False,
        bf16=True,

        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,

        do_eval=False,
        seed=42,
        dataloader_num_workers=0,
        report_to="none",
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    milestone_cb = MilestoneCallback(milestones, snapshot_dir, tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[milestone_cb],
    )

    podcast_sources = [p.stem for p in [LNP_TXT, MINKORREKT_TXT, RAUMZEIT_TXT,
                       FORSCHERGEIST_TXT, CRE_TXT, CCC_CONGRESS_TXT] if p.exists()]

    print(f"\nStarte Training v0.5 für {args.steps:,} Schritte ...")
    print(f"  Effektive Batchgröße: {BATCH_SIZE * GRAD_ACCUM}  (batch={BATCH_SIZE}, accum={GRAD_ACCUM})")
    print(f"  Kontext:              {CONTEXT_LEN} Tokens")
    print(f"  Podcasts aktiv:       {', '.join(podcast_sources) or 'keine'}")
    print(f"  Snapshots bei:        {sorted(milestones)}")

    # Background-Thread: meldet alle 20s den Status während der Compile-Phase
    _compile_done = threading.Event()
    def _monitor():
        t0 = time.time()
        while not _compile_done.wait(20):
            print(f"  [Warte auf Compile... {time.time()-t0:.0f}s] {gpu_stats()}", flush=True)
    threading.Thread(target=_monitor, daemon=True).start()

    resume_from = str(OUTPUT_DIR) if args.resume else None
    trainer.train(resume_from_checkpoint=resume_from)
    _compile_done.set()

    print("\nSpeichere finales Modell ...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"Fertig: {OUTPUT_DIR}")
    print("Nächster Schritt: .venv_ml/bin/python 06_convert_to_gguf.py")


if __name__ == "__main__":
    main()
