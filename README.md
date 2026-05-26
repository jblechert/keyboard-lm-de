# FUTO Keyboard — German Transformer Model

Scripts to train a German-language transformer model for [FUTO Keyboard](https://keyboard.futo.org).
FUTO Keyboard ships an English-only next-word-prediction / autocorrect model; this project builds the German equivalent.

**Pretrained model:** [Releases](https://github.com/jblechert/keyboard-lm-de/releases) — drop the `.gguf` into FUTO Keyboard settings → Language Models.

## Architecture

Matches FUTO's English model:

| Parameter | Value |
|---|---|
| Architecture | Llama (GGUF via llama.cpp) |
| Parameters | ~36M |
| Layers | 8 × 512 hidden dims, 8 attention heads, 1024 FFN |
| Context | 256 tokens |
| Vocabulary | ~15k tokens |
| Tokenizer | SentencePiece BPE, `treat_whitespace_as_suffix=true` |

Special autocorrect tokens: `<XBU>`, `<CHAR_A>`…`<CHAR_Z>`, `<XBC>`, `<XEC>`

## Training data

| Source | Sentences | Weight | Notes |
|---|---|---|---|
| [Tatoeba DE](https://tatoeba.org) | 770k | 3× | Clean everyday German |
| [mC4 DE](https://huggingface.co/datasets/allenai/c4) (allenai/c4) | 5M | 1× | Web text, filtered, ODC-BY |
| Synthetic (Qwen3.6:27b via Ollama) | 2k | 6× | Keyboard-style sentences |

## Pipeline

### 1. Download Tatoeba German sentences → `data/tatoeba_de.txt`
```bash
.venv_ml/bin/python 07_download_tatoeba.py
```

### 2. Stream mC4 German sentences → `data/c4_de.txt`
```bash
.venv_ml/bin/python 09_download_c4_de.py --target 5000000
```

### 3. Generate synthetic keyboard-style sentences → `data/synthetic_de.txt`
```bash
# Requires Ollama running with qwen3.6:27b
.venv_ml/bin/python 08_generate_synthetic.py --target 2000
```

### 4. Train SentencePiece tokenizer → `data/tokenizer/de_keyboard.model`
```bash
.venv_ml/bin/python 04_train_tokenizer.py
```

### 5. Train the model → `data/model_hf/`
```bash
.venv_ml/bin/python 05_train_model.py --steps 54000
# ~15 hours on RX 7900 XTX (ROCm)
```

### 6. Convert to GGUF → `data/de_keyboard.gguf`
```bash
.venv_ml/bin/python 06_convert_to_gguf.py
```

## Requirements

- Python 3.10+
- PyTorch with ROCm (tested: ROCm 7.2, RX 7900 XTX) or CUDA
- `sentencepiece`, `transformers>=4.49,<5`, `tokenizers==0.21.*`, `gguf`, `datasets`

```bash
python -m venv --system-site-packages .venv_ml
.venv_ml/bin/pip install "tokenizers==0.21.0" "transformers>=4.49,<5" datasets gguf sentencepiece
```

## Status

- [x] Training data pipeline (Tatoeba + mC4 + synthetic)
- [x] SentencePiece tokenizer training
- [x] Model training (LlamaForCausalLM, 36M params)
- [x] GGUF conversion + Q4 quantization
- [x] On-device testing with FUTO Keyboard (next-word prediction working)
- [ ] XBU/CHAR autocorrect fine-tuning

## References

- [FUTO Keyboard source](https://github.com/futo-org/android-keyboard)
- [FUTO LM documentation](https://gitlab.futo.org/keyboard/keyboard-wiki/-/wikis/Keyboard-LM-docs)
- [Issue #1212 — Transformer Models for More Languages](https://github.com/futo-org/android-keyboard/issues/1212)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
