#!/usr/bin/env python3
"""
LoRA-Finetuning auf domänenspezifischen Daten.

Lädt das fertige Basismodell und trainiert einen kleinen Adapter nur auf
dem Ziel-Corpus. Die Basisgewichte bleiben eingefroren.

Usage:
  .venv_ml/bin/python 14_finetune_lora.py --domain medizin
  .venv_ml/bin/python 14_finetune_lora.py --domain computer --steps 30000
  .venv_ml/bin/python 14_finetune_lora.py --domain medizin --base-model data/snapshots/step_200000
"""

import argparse
import random
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    DataCollatorForLanguageModeling,
    LlamaForCausalLM,
    LlamaTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import IterableDataset

# ── Pfade ─────────────────────────────────────────────────────────────────────
SP_MODEL    = Path("data/tokenizer/de_keyboard.model")
BASE_MODEL  = Path("data/model_hf")
ADAPTER_DIR = Path("data/adapters")

# ── LoRA-Konfiguration ────────────────────────────────────────────────────────
LORA_R        = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# ── Training ──────────────────────────────────────────────────────────────────
CONTEXT_LEN  = 256
BATCH_SIZE   = 32
GRAD_ACCUM   = 8
LR           = 2e-4
WARMUP_STEPS = 200
LOGGING_STEPS = 100
SAVE_STEPS   = 5_000


def load_tokenizer(base_model_dir: Path) -> LlamaTokenizer:
    tok = LlamaTokenizer(vocab_file=str(SP_MODEL), legacy=False)
    tok.add_special_tokens({
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<unk>",
    })
    return tok


def line_generator(path: Path):
    while True:
        with path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        random.shuffle(lines)
        yield from lines


def make_dataset(sources: list[Path], tokenizer: LlamaTokenizer) -> IterableDataset:
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id

    gens = [line_generator(p) for p in sources]

    def gen():
        buf = []
        while True:
            src = random.choice(gens)
            line = next(src).strip()
            if not line:
                continue
            ids = tokenizer.encode(line, add_special_tokens=False)
            buf += [bos] + ids + [eos]
            while len(buf) >= CONTEXT_LEN:
                yield {"input_ids": buf[:CONTEXT_LEN]}
                buf = buf[CONTEXT_LEN:]

    return IterableDataset.from_generator(gen)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True,
                        help="Domänen-Name (muss data/synthetic_<domain>.txt existieren)")
    parser.add_argument("--steps", type=int, default=20_000,
                        help="Trainingsschritte (default: 20k)")
    parser.add_argument("--base-model", type=Path, default=BASE_MODEL,
                        help=f"Basismodell-Verzeichnis (default: {BASE_MODEL})")
    parser.add_argument("--extra-data", nargs="*", type=Path, default=[],
                        help="Zusätzliche Textdateien für den Finetuning-Corpus")
    args = parser.parse_args()

    domain_file = Path(f"data/synthetic_{args.domain}.txt")
    if not domain_file.exists():
        print(f"Fehler: {domain_file} nicht gefunden.")
        print("Erstelle zuerst mit: .venv_ml/bin/python 12_generate_synthetic_vocab.py "
              f"--topics {args.domain}")
        return

    sources = [domain_file] + args.extra_data
    out_dir = ADAPTER_DIR / args.domain
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Lade Basismodell aus {args.base_model} …")
    tokenizer = load_tokenizer(args.base_model)
    model = LlamaForCausalLM.from_pretrained(
        str(args.base_model),
        torch_dtype=torch.bfloat16,
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    dataset = make_dataset(sources, tokenizer)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        max_steps=args.steps,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        weight_decay=0.01,
        bf16=True,
        gradient_checkpointing=False,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        do_eval=False,
        seed=42,
        dataloader_num_workers=0,
        report_to="none",
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    corpus_names = " + ".join(p.name for p in sources)
    print(f"\nLoRA-Finetuning: {args.domain}")
    print(f"  Corpus:   {corpus_names}")
    print(f"  Schritte: {args.steps:,}")
    print(f"  Adapter → {out_dir}")

    trainer.train()

    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"\nAdapter gespeichert: {out_dir}")
    print(f"Nächster Schritt: .venv_ml/bin/python 15_merge_lora.py --domain {args.domain}")


if __name__ == "__main__":
    main()
