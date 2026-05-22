# FUTO Keyboard — German Transformer Model

Scripts to train a German-language transformer model for [FUTO Keyboard](https://keyboard.futo.org).
FUTO Keyboard ships an English-only next-word-prediction / autocorrect model; this project builds the German equivalent.

## Architecture

The target model matches FUTO's English model:

| Parameter | Value |
|---|---|
| Architecture | Llama (GGUF via llama.cpp) |
| Parameters | ~36M |
| Layers | 8 × 512 hidden dims |
| Attention heads | 8 |
| Vocabulary | 15,008 tokens |
| Tokenizer | SentencePiece, `treat_whitespace_as_suffix=true` |

Special autocorrect tokens: `<XBU>`, `<CHAR_A>`…`<CHAR_Z>`, `<XBC>`, `<XEC>`

## Requirements

- Python 3.10+
- PyTorch with ROCm (tested: ROCm 7.2, RX 7900 XTX) or CUDA
- System packages: `sentencepiece`, `transformers>=4.49,<5`, `tokenizers==0.21.*`, `gguf`

```bash
python -m venv --system-site-packages .venv_ml
.venv_ml/bin/pip install "tokenizers==0.21.0" "transformers>=4.49,<5"
```

## Pipeline

### 1. Download German Wikipedia dump (~7 GB)
```bash
./01_download.sh
```

### 2. Extract and clean plain text → `data/train.txt`
```bash
.venv/bin/python 02_extract.py
# ~30 min, produces one sentence per line
```

### 3. Train SentencePiece tokenizer → `data/tokenizer/de_keyboard.model`
```bash
.venv_ml/bin/python 04_train_tokenizer.py
```

### 4. Sanity-check the tokenizer
```bash
.venv_ml/bin/python 04b_check_tokenizer.py
# Exit 0 = all checks passed
```

### 5. Train the model *(coming soon)*
```bash
.venv_ml/bin/python 05_train_model.py
```

### 6. Convert to GGUF → `data/de_keyboard.gguf`
```bash
.venv_ml/bin/python 06_convert_to_gguf.py --quantize f16
```

## Status

- [x] Data pipeline (Wikipedia download + extraction + cleaning)
- [x] SentencePiece tokenizer training
- [x] Tokenizer sanity checks
- [x] GGUF conversion
- [ ] Model training script (`05_train_model.py`)
- [ ] Autocorrect fine-tuning (misspelling dataset)
- [ ] On-device testing with FUTO Keyboard

## References

- [FUTO Keyboard source](https://gitlab.futo.org/keyboard/latinime)
- [FUTO LM documentation](https://gitlab.futo.org/keyboard/keyboard-wiki/-/wikis/Keyboard-LM-docs)
- [Issue #1212 — Transformer Models for More Languages](https://github.com/futo-org/android-keyboard/issues/1212)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
