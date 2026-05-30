#!/usr/bin/env python3
"""
Trainiert einen LoRA-Adapter auf dem FUTO-DE Basismodell, spezialisiert auf
CCC Congress-Talks (Netzpolitik, Kryptographie, Hacker-Kultur, Tech-Gesellschaft).

Der Adapter kann separat gespeichert und bei Bedarf auf das Basismodell
aufgeschaltet werden — ohne das Basismodell zu verändern.

Voraussetzungen:
  - Basismodell trainiert: models/de_keyboard/
  - Congress-Daten heruntergeladen: data/ccc_congress_de.txt
  - pip install peft

Output:
  data/lora_congress/   LoRA-Gewichte (~wenige MB)

Usage:
  .venv_ml/bin/python 22_train_congress_lora.py [--steps 5000] [--rank 16]
"""

import argparse
import random
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import LlamaForCausalLM, LlamaConfig, PreTrainedTokenizerFast
import sentencepiece as spm

SP_MODEL   = Path("data/tokenizer/de_keyboard.model")
BASE_MODEL = Path("models/de_keyboard")
CORPUS     = Path("data/ccc_congress_de.txt")
OUTPUT_DIR = Path("data/lora_congress")

CONTEXT    = 256
BATCH_SIZE = 16
LR         = 2e-4


def load_tokenizer():
    sp = spm.SentencePieceProcessor()
    sp.Load(str(SP_MODEL))
    return sp


def token_stream(sp, path: Path, context: int):
    """Endloser Token-Stream aus dem Corpus."""
    lines = path.read_text(encoding="utf-8").splitlines()
    buf = []
    while True:
        random.shuffle(lines)
        for line in lines:
            ids = sp.Encode(line, out_type=int)
            buf.extend(ids + [sp.eos_id()])
            while len(buf) >= context + 1:
                yield buf[:context + 1]
                buf = buf[context + 1:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--rank",  type=int, default=16,
                        help="LoRA rank (höher = mehr Kapazität, größere Datei)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"LoRA rank: {args.rank}  |  Steps: {args.steps}")

    if not CORPUS.exists():
        print(f"Corpus nicht gefunden: {CORPUS}")
        print("Erst 21_download_ccc_congress.py ausführen.")
        return

    sp = load_tokenizer()
    vocab_size = sp.GetPieceSize()

    # Basismodell laden
    print("Lade Basismodell …")
    model = LlamaForCausalLM.from_pretrained(str(BASE_MODEL))
    model = model.to(device)

    # LoRA-Adapter aufschalten
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    stream = token_stream(sp, CORPUS, CONTEXT)
    model.train()

    print(f"\nTrainiere {args.steps} Steps …\n")
    loss_acc = 0.0
    for step in range(1, args.steps + 1):
        batch = [next(stream) for _ in range(BATCH_SIZE)]
        ids = torch.tensor(batch, dtype=torch.long, device=device)
        input_ids = ids[:, :-1]
        labels    = ids[:, 1:].clone()

        optimizer.zero_grad()
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_acc += out.loss.item()
        if step % 100 == 0:
            print(f"  Step {step:5d}/{args.steps}  loss={loss_acc/100:.4f}")
            loss_acc = 0.0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR))
    print(f"\nLoRA-Adapter gespeichert: {OUTPUT_DIR}/")
    print("Laden später mit:")
    print("  from peft import PeftModel")
    print(f"  model = PeftModel.from_pretrained(base_model, '{OUTPUT_DIR}')")


if __name__ == "__main__":
    main()
