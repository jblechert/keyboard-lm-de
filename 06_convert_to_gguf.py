#!/usr/bin/env python3
"""
Converts a trained HuggingFace Llama model + SentencePiece tokenizer
into a GGUF file compatible with FUTO Keyboard.

Input:
  data/model_hf/                     HuggingFace model directory (from 05_train_model.py)
  data/tokenizer/de_keyboard.model   SentencePiece model (from 04_train_tokenizer.py)

Output:
  data/de_keyboard.gguf

Usage:
  .venv_ml/bin/python 06_convert_to_gguf.py [--quantize q8_0]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from gguf import GGUFWriter, GGMLQuantizationType, TokenType

MODEL_HF_DIR = Path("data/model_hf")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantize", default="f16", choices=QUANT_MAP.keys(),
                        help="Weight quantization type (default: f16)")
    parser.add_argument("--model-dir", default=str(MODEL_HF_DIR))
    parser.add_argument("--sp-model",  default=str(SP_MODEL))
    parser.add_argument("--output",    default=str(OUT_GGUF))
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    sp_path   = Path(args.sp_model)
    out_path  = Path(args.output)

    for p, label in [(model_dir, "HF-Modell"), (sp_path, "SP-Tokenizer")]:
        if not p.exists():
            print(f"Fehler: {label} nicht gefunden: {p}", file=sys.stderr)
            sys.exit(1)

    # ── Modell laden ──────────────────────────────────────────────────────────
    print("Lade HuggingFace-Modell …")
    config = LlamaConfig.from_pretrained(str(model_dir))
    model  = LlamaForCausalLM.from_pretrained(str(model_dir), torch_dtype=torch.float32)
    state  = model.state_dict()

    # ── Tokenizer laden ───────────────────────────────────────────────────────
    print("Lade SentencePiece-Tokenizer …")
    tokens, scores, types, vocab_size = load_tokenizer_vocab(sp_path)

    sp_bytes = sp_path.read_bytes()   # raw protobuf — wird im GGUF eingebettet

    # ── GGUF schreiben ────────────────────────────────────────────────────────
    print(f"Schreibe GGUF → {out_path} …")
    writer = GGUFWriter(str(out_path), arch="llama")

    # Modell-Metadaten
    writer.add_name("mjb-de-0.1")
    writer.add_description("German keyboard language model for FUTO Keyboard")
    writer.add_languages(["de"])

    # Llama-Architektur (muss zu config passen)
    writer.add_context_length(config.max_position_embeddings)
    writer.add_embedding_length(config.hidden_size)
    writer.add_block_count(config.num_hidden_layers)
    writer.add_feed_forward_length(config.intermediate_size)
    writer.add_head_count(config.num_attention_heads)
    writer.add_head_count_kv(config.num_key_value_heads)
    writer.add_layer_norm_rms_eps(config.rms_norm_eps)
    writer.add_rope_freq_base(getattr(config, "rope_theta", 10000.0))
    writer.add_file_type(QUANT_MAP[args.quantize])

    # Standard-GGUF-Tokenizer-Metadaten (für llama.cpp-Vokabular-Lookup)
    writer.add_tokenizer_model("llama")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(types)
    writer.add_bos_token_id(config.bos_token_id or 1)
    writer.add_eos_token_id(config.eos_token_id or 2)
    writer.add_unk_token_id(0)
    writer.add_pad_token_id(3)

    # FUTO-spezifische Metadaten (keyboardlm.* Keys — siehe ModelMeta.cpp)
    writer.add_string("keyboardlm.languages", "de")
    writer.add_string("keyboardlm.features", "xbu_char_autocorrect_v1 char_embed_mixing_v1")
    writer.add_string("keyboardlm.ext_tokenizer_type", "sentencepiece")
    # SentencePiece-Protobuf als UINT8-Array einbetten (LoadFromSerializedProto erwartet das)
    writer.add_array("keyboardlm.ext_tokenizer_data", sp_bytes)

    # Gewichte
    print(f"Konvertiere {len(state)} Tensoren …")
    for hf_name, tensor in state.items():
        gguf_name = HF_TO_GGUF.get(hf_name)
        if gguf_name is None:
            print(f"  [skip] {hf_name}")
            continue

        np_tensor = tensor.float().numpy()
        # Norm weights must stay F32 regardless of quantization
        if gguf_name.endswith("_norm.weight"):
            np_quant, ggml_type = np_tensor.astype(np.float32), GGMLQuantizationType.F32
        else:
            np_quant, ggml_type = quantize_tensor(np_tensor, args.quantize)
        writer.add_tensor(gguf_name, np_quant, raw_dtype=ggml_type)
        print(f"  {hf_name} → {gguf_name}  {list(np_tensor.shape)}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nFertig: {out_path}  ({size_mb:.0f} MB)")
    print("Tipp: für echte Q4/Q8-Quantisierung anschließend llama.cpp's")
    print("      quantize-Tool ausführen: ./quantize de_keyboard.gguf q4_0")


if __name__ == "__main__":
    main()
