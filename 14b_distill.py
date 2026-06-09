#!/usr/bin/env python3
"""
Trainiert ein schlankes Student-Modell via Knowledge Distillation.

Der Teacher (z.B. v0.5.1 @ 35k, 136M) läuft parallel im eval()-Modus und
liefert Soft-Targets; der Student minimiert:

    loss = alpha * KL(student || teacher)  +  (1 - alpha) * CE(student, labels)

Der KL-Term wird mit Temperatur T skaliert (T² kompensiert den Normierungs-
effekt), was dem Student erlaubt, die "dunkle Seite" der Teacher-Verteilung zu lernen.

Usage:
  .venv_ml/bin/python 14b_distill.py \\
      --teacher models/de_keyboard/checkpoint-35000

  .venv_ml/bin/python 14b_distill.py \\
      --teacher models/de_keyboard/checkpoint-35000 \\
      --student-size slim \\
      --steps 100000 --alpha 0.7 --temperature 4.0
"""

import argparse
import math
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
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

# ── Pfade (gleich wie 05_train_model.py) ─────────────────────────────────────
SP_MODEL            = Path("data/tokenizer/de_keyboard.model")
TATOEBA_TXT         = Path("data/tatoeba_de.txt")
C4_TXT              = Path("data/c4_de.txt")
FINEWEB2_TXT        = Path("data/fineweb2_de.txt")
SYNTHETIC_GLOB      = "data/synthetic_*.txt"
PRIVATE_TXT         = Path("data/private_de.txt")
PARLAMENTSREVUE_TXT = Path("data/parlamentsrevue_de.txt")
LNP_TXT             = Path("data/lnp_de.txt")
MINKORREKT_TXT      = Path("data/minkorrekt_de.txt")
RAUMZEIT_TXT        = Path("data/raumzeit_de.txt")
FORSCHERGEIST_TXT   = Path("data/forschergeist_de.txt")
CRE_TXT             = Path("data/cre_de.txt")
CCC_CONGRESS_TXT    = Path("data/ccc_congress_de.txt")

TATOEBA_WEIGHT      = 3
C4_WEIGHT           = 1
FINEWEB2_WEIGHT     = 1
SYNTHETIC_WEIGHT    = 3
PRIVATE_WEIGHT      = 2
PARLAMENTSREVUE_WEIGHT = 2
PODCAST_WEIGHT      = 2

# ── Student-Architekturen ────────────────────────────────────────────────────
STUDENT_CONFIGS = {
    # v0.5-Größenklasse: gleiche Parameter wie der Vorgänger vor der 136M-Erweiterung
    "slim": dict(
        hidden_size=512,
        num_hidden_layers=10,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=256,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        attention_bias=False,
        hidden_act="silu",
    ),
    # Noch kompakter für sehr eingeschränkte Geräte (~12M Params)
    "tiny": dict(
        hidden_size=384,
        num_hidden_layers=6,
        num_attention_heads=6,
        intermediate_size=1024,
        max_position_embeddings=256,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        attention_bias=False,
        hidden_act="silu",
    ),
}

# ── Training-Hyperparameter ───────────────────────────────────────────────────
CONTEXT_LEN     = 256
BATCH_SIZE      = 64
GRAD_ACCUM      = 4
LR              = 3e-4
LR_WARMUP_STEPS = 1000
WEIGHT_DECAY    = 0.1
MAX_GRAD_NORM   = 1.0
SAVE_STEPS      = 5_000
LOGGING_STEPS   = 200


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


# ── Daten (identisch mit 05_train_model.py) ──────────────────────────────────

def _has_content(path: Path) -> bool:
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


def line_generator(path: Path, shuffle_buffer: int = 50_000):
    while True:
        buf = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                buf.append(line)
                if len(buf) >= shuffle_buffer:
                    random.shuffle(buf)
                    yield from buf
                    buf = []
        if buf:
            random.shuffle(buf)
            yield from buf


def mixed_generator():
    sources = []
    if _has_content(TATOEBA_TXT):
        sources.append((line_generator(TATOEBA_TXT), TATOEBA_WEIGHT))
    if _has_content(C4_TXT):
        sources.append((line_generator(C4_TXT), C4_WEIGHT))
    if _has_content(FINEWEB2_TXT):
        sources.append((line_generator(FINEWEB2_TXT), FINEWEB2_WEIGHT))
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
    thresholds, acc = [], 0
    for w in weights:
        acc += w / total
        thresholds.append(acc)

    while True:
        r = random.random()
        for gen, threshold in zip(gens, thresholds):
            if r < threshold:
                yield next(gen).strip()
                break


def tokenize_and_chunk(tokenizer: LlamaTokenizer) -> IterableDataset:
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id

    def gen():
        buf = []
        for line in mixed_generator():
            if not line:
                continue
            buf += [bos] + tokenizer.encode(line, add_special_tokens=False) + [eos]
            while len(buf) >= CONTEXT_LEN:
                yield {"input_ids": buf[:CONTEXT_LEN]}
                buf = buf[CONTEXT_LEN:]

    return IterableDataset.from_generator(gen)


# ── Distillation-Trainer ──────────────────────────────────────────────────────

class DistillTrainer(Trainer):
    """HF-Trainer mit KL-Distillation-Loss gegen einen fixierten Teacher."""

    def __init__(self, teacher: LlamaForCausalLM, alpha: float, temperature: float, **kwargs):
        super().__init__(**kwargs)
        self.teacher     = teacher
        self.alpha       = alpha
        self.temperature = temperature

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]                        # [B, T]
        B, T = input_ids.shape
        V = model.config.vocab_size

        # Student forward (kein Label-Shift hier — machen wir manuell)
        student_out    = model(input_ids=input_ids)
        student_logits = student_out.logits                    # [B, T, V]

        # Teacher forward — kein Gradient, kein Optimizer-State
        with torch.no_grad():
            teacher_logits = self.teacher(input_ids=input_ids).logits  # [B, T, V]

        # Shift: Position i sagt Token i+1 vorher
        s = student_logits[:, :-1, :].contiguous().view(-1, V)   # [(B·T-1), V]
        t = teacher_logits[:, :-1, :].contiguous().view(-1, V)
        labels = input_ids[:, 1:].contiguous().view(-1)           # [(B·T-1)]

        # Cross-Entropy gegen Ground-Truth
        ce_loss = F.cross_entropy(s, labels)

        # KL-Divergenz gegen Teacher-Softmax (mit Temperatur)
        T_scale = self.temperature
        s_log_p = F.log_softmax(s / T_scale, dim=-1)
        t_p     = F.softmax(t / T_scale, dim=-1)
        # T² normiert die reduzierte Magnitude durch Temperature-Skalierung
        kl_loss = F.kl_div(s_log_p, t_p, reduction="batchmean") * (T_scale ** 2)

        loss = self.alpha * kl_loss + (1.0 - self.alpha) * ce_loss

        return (loss, student_out) if return_outputs else loss


# ── Logging-Callback ──────────────────────────────────────────────────────────

class LogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.global_step > 0:
            loss = logs.get("loss", "?")
            lr   = logs.get("learning_rate", "?")
            if isinstance(loss, float):
                loss = f"{loss:.4f}"
            if isinstance(lr, float):
                lr = f"{lr:.2e}"
            print(f"  Step {state.global_step:6d} | loss={loss} lr={lr}", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation: Teacher → schlanker Student"
    )
    parser.add_argument("--teacher",      required=True,
                        help="Pfad zum Teacher-Modell (HF-Format), z.B. models/de_keyboard/checkpoint-35000")
    parser.add_argument("--student-size", default="slim", choices=list(STUDENT_CONFIGS),
                        help="Student-Architektur (default: slim ~36M)")
    parser.add_argument("--output",       default="models/de_keyboard_v0.5.2",
                        help="Ausgabeverzeichnis für den Student")
    parser.add_argument("--steps",        type=int,   default=100_000)
    parser.add_argument("--alpha",        type=float, default=0.7,
                        help="Gewicht des KL-Terms (0=nur CE, 1=nur Distillation, default: 0.7)")
    parser.add_argument("--temperature",  type=float, default=4.0,
                        help="Softening-Temperatur für Teacher-Verteilung (default: 4.0)")
    parser.add_argument("--resume",       action="store_true")
    args = parser.parse_args()

    teacher_path = Path(args.teacher)
    output_dir   = Path(args.output)

    if not SP_MODEL.exists():
        print(f"Fehler: Tokenizer nicht gefunden: {SP_MODEL}")
        return
    if not teacher_path.exists():
        print(f"Fehler: Teacher-Modell nicht gefunden: {teacher_path}")
        return

    print("Lade Tokenizer …")
    tokenizer = load_tokenizer()

    # ── Teacher laden (bf16, eval, kein Grad) ────────────────────────────────
    print(f"Lade Teacher: {teacher_path} …")
    teacher = LlamaForCausalLM.from_pretrained(str(teacher_path), torch_dtype=torch.bfloat16)
    teacher = teacher.cuda().eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    t_params = sum(p.numel() for p in teacher.parameters())
    print(f"  Teacher: {t_params/1e6:.1f}M Parameter  (frozen)")

    # ── Student bauen oder laden ──────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and (output_dir / "config.json").exists():
        print(f"Lade Student-Checkpoint: {output_dir} …")
        student = LlamaForCausalLM.from_pretrained(str(output_dir))
    else:
        print(f"Baue neuen Student ({args.student_size}) …")
        cfg = LlamaConfig(
            vocab_size=len(tokenizer),
            **STUDENT_CONFIGS[args.student_size],
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        student = LlamaForCausalLM(cfg)
    s_params = sum(p.numel() for p in student.parameters())
    print(f"  Student: {s_params/1e6:.1f}M Parameter")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("Erstelle Dataset …")
    dataset  = tokenize_and_chunk(tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ── Training-Args ─────────────────────────────────────────────────────────
    train_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=args.steps,

        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,

        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=LR_WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        optim="adamw_torch",

        bf16=True,
        gradient_checkpointing=False,

        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,

        do_eval=False,
        seed=42,
        dataloader_num_workers=0,
        report_to="none",
    )

    trainer = DistillTrainer(
        teacher=teacher,
        alpha=args.alpha,
        temperature=args.temperature,
        model=student,
        args=train_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[LogCallback()],
    )

    print(f"\nStarte Distillation …")
    print(f"  Teacher:      {teacher_path}  ({t_params/1e6:.1f}M)")
    print(f"  Student:      {args.student_size}  ({s_params/1e6:.1f}M)")
    print(f"  alpha:        {args.alpha}  (KL-Gewicht)")
    print(f"  temperature:  {args.temperature}")
    print(f"  Steps:        {args.steps:,}")
    print(f"  Output:       {output_dir}")

    resume_from = str(output_dir) if args.resume else None
    trainer.train(resume_from_checkpoint=resume_from)

    print("\nSpeichere Student …")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Fertig: {output_dir}")
    print(f"Nächster Schritt: .venv_ml/bin/python 06_convert_to_gguf.py --model-dir {output_dir} --version v0.5.2")


if __name__ == "__main__":
    main()
