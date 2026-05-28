#!/usr/bin/env python3
"""
Merged einen LoRA-Adapter in das Basismodell und konvertiert zu GGUF.

Das Ergebnis ist ein normales GGUF exakt gleicher Größe wie das Basismodell —
mit domänenspezifisch angepassten Gewichten. Kein Adapter-Loading zur Laufzeit nötig.

Usage:
  .venv_ml/bin/python 15_merge_lora.py --domain medizin
  .venv_ml/bin/python 15_merge_lora.py --domain computer --no-gguf
  .venv_ml/bin/python 15_merge_lora.py --domain medizin --base-model data/snapshots/step_200000
"""

import argparse
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer

# ── Pfade ─────────────────────────────────────────────────────────────────────
SP_MODEL    = Path("data/tokenizer/de_keyboard.model")
BASE_MODEL  = Path("data/model_hf")
ADAPTER_DIR = Path("data/adapters")
MERGED_DIR  = Path("data/model_merged")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True,
                        help="Domänen-Name (muss data/adapters/<domain>/ existieren)")
    parser.add_argument("--base-model", type=Path, default=BASE_MODEL,
                        help=f"Basismodell-Verzeichnis (default: {BASE_MODEL})")
    parser.add_argument("--no-gguf", action="store_true",
                        help="Nur mergen, kein GGUF erzeugen")
    parser.add_argument("--no-quantize", action="store_true",
                        help="Nur F16-GGUF, keine Quantisierungen")
    args = parser.parse_args()

    adapter_dir = ADAPTER_DIR / args.domain
    if not adapter_dir.exists():
        print(f"Fehler: Adapter nicht gefunden: {adapter_dir}")
        print(f"Erstelle zuerst mit: .venv_ml/bin/python 14_finetune_lora.py "
              f"--domain {args.domain}")
        sys.exit(1)

    out_dir = MERGED_DIR / args.domain
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Lade Basismodell aus {args.base_model} …")
    tok = LlamaTokenizer(vocab_file=str(SP_MODEL), legacy=False)
    tok.add_special_tokens({
        "bos_token": "<s>", "eos_token": "</s>",
        "unk_token": "<unk>", "pad_token": "<unk>",
    })

    base = LlamaForCausalLM.from_pretrained(
        str(args.base_model),
        torch_dtype=torch.bfloat16,
    )

    print(f"Lade Adapter aus {adapter_dir} …")
    model = PeftModel.from_pretrained(base, str(adapter_dir))

    print("Merge Adapter in Basisgewichte …")
    model = model.merge_and_unload()

    print(f"Speichere gemergtes Modell → {out_dir}")
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print("Merge abgeschlossen.")

    if args.no_gguf:
        print(f"\nKein GGUF angefordert. Modell liegt in: {out_dir}")
        return

    print(f"\nStarte GGUF-Konvertierung für Domäne '{args.domain}' …")
    cmd = [
        sys.executable, "06_convert_to_gguf.py",
        "--model-dir", str(out_dir),
        "--name", f"mjb-de-{args.domain}",
    ]
    if args.no_quantize:
        cmd.append("--no-quantize")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Fehler bei GGUF-Konvertierung.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
