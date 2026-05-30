#!/usr/bin/env python3
"""
Converts a trained HuggingFace Llama model + SentencePiece tokenizer
into a GGUF file compatible with FUTO Keyboard.

Input:
  models/de_keyboard/                     HuggingFace model directory (from 05_train_model.py)
  data/tokenizer/de_keyboard.model   SentencePiece model (from 04_train_tokenizer.py)

Output:
  data/de_keyboard.gguf

Usage:
  .venv_ml/bin/python 06_convert_to_gguf.py
  .venv_ml/bin/python 06_convert_to_gguf.py --all-snapshots   # konvertiert data/snapshots/step_*/
  .venv_ml/bin/python 06_convert_to_gguf.py --no-quantize     # nur F16
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from gguf import GGUFWriter, GGMLQuantizationType, TokenType

MODEL_HF_DIR = Path("models/de_keyboard")
SP_MODEL     = Path("data/tokenizer/de_keyboard.model")
OUT_GGUF     = Path("data/de_keyboard.gguf")

# FUTO special tokens (must match 04_train_tokenizer.py)
FUTO_SPECIAL_TOKENS = [
    "<XBU>", "<XBC>", "<XEC>",
    *[f"<CHAR_{chr(c)}>" for c in range(ord("A"), ord("Z") + 1)],
]

# HuggingFace tensor name → GGUF tensor name
HF_TO_GGUF = {
    "model.embed_tokens.weight":  "token_embd.weight",
    "model.norm.weight":          "output_norm.weight",
    "lm_head.weight":             "output.weight",
}
for i in range(64):  # enough layers for any reasonable model
    p = f"model.layers.{i}"
    g = f"blk.{i}"
    HF_TO_GGUF.update({
        f"{p}.self_attn.q_proj.weight":             f"{g}.attn_q.weight",
        f"{p}.self_attn.k_proj.weight":             f"{g}.attn_k.weight",
        f"{p}.self_attn.v_proj.weight":             f"{g}.attn_v.weight",
        f"{p}.self_attn.o_proj.weight":             f"{g}.attn_output.weight",
        f"{p}.mlp.gate_proj.weight":                f"{g}.ffn_gate.weight",
        f"{p}.mlp.down_proj.weight":                f"{g}.ffn_down.weight",
        f"{p}.mlp.up_proj.weight":                  f"{g}.ffn_up.weight",
        f"{p}.input_layernorm.weight":              f"{g}.attn_norm.weight",
        f"{p}.post_attention_layernorm.weight":     f"{g}.ffn_norm.weight",
    })

QUANT_MAP = {
    "f32":  GGMLQuantizationType.F32,
    "f16":  GGMLQuantizationType.F16,
    "q8_0": GGMLQuantizationType.Q8_0,
    "q4_0": GGMLQuantizationType.Q4_0,
}


def load_tokenizer_vocab(sp_model_path: Path):
    """Extract token strings, scores, and types from a SentencePiece model."""
    sp = spm.SentencePieceProcessor(model_file=str(sp_model_path))
    vocab_size = sp.get_piece_size()

    tokens, scores, types = [], [], []
    for i in range(vocab_size):
        piece = sp.id_to_piece(i)
        score = sp.get_score(i)
        tokens.append(piece.encode("utf-8"))
        scores.append(score)

        if sp.IsUnknown(i):
            types.append(TokenType.UNKNOWN)
        elif sp.IsControl(i):
            types.append(TokenType.CONTROL)
        elif sp.IsByte(i):
            types.append(TokenType.BYTE)
        else:
            types.append(TokenType.NORMAL)

    return tokens, scores, types, vocab_size


def quantize_tensor(tensor: np.ndarray, quant: str) -> tuple[np.ndarray, GGMLQuantizationType]:
    """Return (quantized_array, ggml_type). Only f32/f16 implemented here;
    Q4/Q8 quantization needs llama.cpp for production use."""
    if quant == "f16":
        return tensor.astype(np.float16), GGMLQuantizationType.F16
    elif quant in ("q8_0", "q4_0"):
        # Fallback: use f16 — for true INT quantization pipe through llama.cpp's quantize tool
        print(f"  [warn] {quant} quantization not implemented here, falling back to f16")
        return tensor.astype(np.float16), GGMLQuantizationType.F16
    return tensor.astype(np.float32), GGMLQuantizationType.F32


import subprocess

# Quantization levels produced by default after F16 conversion
DEFAULT_QUANTS = ["Q4_0", "Q6_K", "Q8_0"]


def run_quantize(src: Path, dst: Path, quant: str):
    result = subprocess.run(
        ["llama-quantize", str(src), str(dst), quant],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  [fehler] llama-quantize: {result.stderr.strip()}", file=sys.stderr)
        return False
    for line in result.stdout.splitlines():
        if "quant size" in line or "total time" in line:
            print(f"  {line.strip()}")
    return True


SNAPSHOT_DIR = Path("data/snapshots")


def convert_one(model_dir: Path, sp_path: Path, out_path: Path,
                name: str, no_quantize: bool):
    """Konvertiert ein einzelnes HF-Modell zu GGUF (F16 + optional quantisierte Varianten)."""
    # ── Modell laden ──────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Konvertiere: {model_dir}  →  {out_path}")
    print("Lade HuggingFace-Modell …")
    config = LlamaConfig.from_pretrained(str(model_dir))
    model  = LlamaForCausalLM.from_pretrained(str(model_dir), torch_dtype=torch.float32)
    state  = model.state_dict()

    # ── Tokenizer laden ───────────────────────────────────────────────────────
    print("Lade SentencePiece-Tokenizer …")
    tokens, scores, types, vocab_size = load_tokenizer_vocab(sp_path)
    sp_bytes = sp_path.read_bytes()

    # ── F16 GGUF schreiben ────────────────────────────────────────────────────
    print(f"Schreibe F16-GGUF → {out_path} …")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(str(out_path), arch="llama")

    writer.add_name(name)
    writer.add_description("German keyboard language model for FUTO Keyboard")
    writer.add_languages(["de"])

    writer.add_context_length(config.max_position_embeddings)
    writer.add_embedding_length(config.hidden_size)
    writer.add_block_count(config.num_hidden_layers)
    writer.add_feed_forward_length(config.intermediate_size)
    writer.add_head_count(config.num_attention_heads)
    writer.add_head_count_kv(config.num_key_value_heads)
    writer.add_layer_norm_rms_eps(config.rms_norm_eps)
    writer.add_rope_freq_base(getattr(config, "rope_theta", 10000.0))
    writer.add_file_type(GGMLQuantizationType.F16)

    writer.add_tokenizer_model("llama")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(types)
    writer.add_bos_token_id(config.bos_token_id or 1)
    writer.add_eos_token_id(config.eos_token_id or 2)
    writer.add_unk_token_id(0)
    writer.add_pad_token_id(3)

    writer.add_string("keyboardlm.languages", "de")
    writer.add_string("keyboardlm.features", "xbu_char_autocorrect_v1 char_embed_mixing_v1")
    writer.add_string("keyboardlm.ext_tokenizer_type", "sentencepiece")
    writer.add_array("keyboardlm.ext_tokenizer_data", sp_bytes)

    print(f"Konvertiere {len(state)} Tensoren …")
    for hf_name, tensor in state.items():
        gguf_name = HF_TO_GGUF.get(hf_name)
        if gguf_name is None:
            continue
        np_tensor = tensor.float().numpy()
        if gguf_name.endswith("_norm.weight"):
            np_quant, ggml_type = np_tensor.astype(np.float32), GGMLQuantizationType.F32
        else:
            np_quant, ggml_type = np_tensor.astype(np.float16), GGMLQuantizationType.F16
        writer.add_tensor(gguf_name, np_quant, raw_dtype=ggml_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  → {out_path}  ({size_mb:.0f} MB)")

    # ── Quantisierte Varianten via llama-quantize ─────────────────────────────
    if no_quantize:
        return

    stem   = out_path.stem.removesuffix("-f16").removesuffix("-F16")
    parent = out_path.parent
    print("Quantisiere mit llama-quantize …")
    for quant in DEFAULT_QUANTS:
        dst = parent / f"{stem}-{quant}.gguf"
        print(f"  {quant} → {dst.name}")
        if run_quantize(out_path, dst, quant):
            size_mb = dst.stat().st_size / 1024 / 1024
            print(f"  → {size_mb:.0f} MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",      default=None,
                        help="Model name embedded in GGUF (overrides auto-generated name)")
    parser.add_argument("--version",   default="v0.4",
                        help="Model version string (default: v0.4)")
    parser.add_argument("--model-dir", default=str(MODEL_HF_DIR))
    parser.add_argument("--sp-model",  default=str(SP_MODEL))
    parser.add_argument("--output",    default=str(OUT_GGUF),
                        help="Output path for F16 base GGUF (quantized variants derived from name)")
    parser.add_argument("--no-quantize", action="store_true",
                        help="Skip llama-quantize step (F16 only)")
    parser.add_argument("--all-snapshots", action="store_true",
                        help="Konvertiert alle Snapshots in data/snapshots/<version>/step_*/")
    args = parser.parse_args()

    sp_path = Path(args.sp_model)
    if not sp_path.exists():
        print(f"Fehler: SP-Tokenizer nicht gefunden: {sp_path}", file=sys.stderr)
        sys.exit(1)

    if args.all_snapshots:
        snap_dir = Path("data/snapshots") / args.version
        snapshots = sorted(snap_dir.glob("step_*"))
        if not snapshots:
            print(f"Keine Snapshots in {snap_dir}/", file=sys.stderr)
            sys.exit(1)
        print(f"{len(snapshots)} Snapshots gefunden: {[s.name for s in snapshots]}")
        for snap in snapshots:
            step_str = snap.name.replace("step_", "").lstrip("0") or "0"
            step_k   = int(step_str) // 1000
            name     = f"mjb-de-{args.version}-{step_k}k"
            out_path = Path("data") / f"{name}.gguf"
            convert_one(snap, sp_path, out_path, name, args.no_quantize)
        print("\nAlle Snapshots konvertiert.")
        return

    model_dir = Path(args.model_dir)
    out_path  = Path(args.output)

    if not model_dir.exists():
        print(f"Fehler: HF-Modell nicht gefunden: {model_dir}", file=sys.stderr)
        sys.exit(1)

    name = args.name or f"mjb-de-{args.version}"
    convert_one(model_dir, sp_path, out_path, name, args.no_quantize)
    print("\nFertig.")


if __name__ == "__main__":
    main()
