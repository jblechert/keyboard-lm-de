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
- PyTorch with CUDA (any NVIDIA GPU) or ROCm (AMD, Linux only)
- `sentencepiece`, `transformers>=4.49,<5`, `tokenizers==0.21.*`, `gguf`, `datasets`

```bash
python -m venv --system-site-packages .venv_ml
.venv_ml/bin/pip install "tokenizers==0.21.0" "transformers>=4.49,<5" datasets gguf sentencepiece
```

### Hardware

The 36M-parameter model is small — it uses **~1–2 GB VRAM** even at full batch size.
Any discrete GPU with 4 GB+ VRAM can train it; the bottleneck is speed, not memory.

| GPU | VRAM | 50k steps | 100k steps |
|-----|------|-----------|------------|
| RX 7900 XTX / RTX 4090 | 24 GB | ~14 h | ~28 h |
| RTX 3080 / RTX 4070 (10–12 GB) | 10–12 GB | ~18 h | ~36 h |
| RTX 3060 12 GB | 12 GB | ~38 h | ~75 h |
| GTX 1080 Ti | 11 GB | ~30 h | ~60 h |

> **bf16**: The training script uses `bf16=True`, which requires NVIDIA Ampere (RTX 30xx+) or AMD ROCm.
> For older NVIDIA cards (GTX 10xx/20xx), change `bf16=True` → `fp16=True` in `05_train_model.py`.

## On-device performance

### Model size vs. training steps — two independent dimensions

These are often confused:

| Dimension | What it affects | What it does NOT affect |
|-----------|----------------|------------------------|
| **Parameters** (36M / 55M / 72M) | Inference speed and RAM on device | Prediction quality |
| **Training steps** (54k / 100k / 200k) | Prediction quality | Inference speed or RAM |

In practice: a 36M model trained for 200k steps runs **identically fast** on your phone as the same 36M model at 54k steps — it just predicts better. A 72M model at 54k steps would be ~2× slower than a 36M model at 200k steps, regardless of training time.

### Quantization tiers

GGUF supports multiple quantization levels that trade file size / RAM against precision.
For v0.3 we plan to ship five tiers (sizes for the 36M model):

| Tier | Quantization | File size | Quality loss | Target devices |
|------|-------------|-----------|-------------|----------------|
| **ultra-low** | Q2_K | ~11 MB | noticeable | very old / low-RAM phones |
| **low** | Q4_K_S | ~18 MB | minimal | mid-range |
| **medium** | Q4_K_M | ~20 MB | minimal | mid-range |
| **high** | Q6_K | ~24 MB | near-zero | flagship |
| **ultra-high** | Q8_0 | ~36 MB | effectively none | flagship, confirmed working on Moto Edge 40 Pro |

Q4_K_M is the recommended default for most devices.

### Device benchmarks

Benchmarks are measured with `17_benchmark_device.py` via ADB.
Reference device: **Motorola Edge 40 Pro** (benchmark score to be confirmed on device).

> Results will be published with the v0.3 release.

## Status

- [x] Training data pipeline (Tatoeba + mC4 + synthetic)
- [x] SentencePiece tokenizer training
- [x] Model training (LlamaForCausalLM, 36M params)
- [x] GGUF conversion + Q4 quantization
- [x] On-device testing with FUTO Keyboard (next-word prediction working)
- [ ] XBU/CHAR autocorrect fine-tuning

## License

**Code** (this repository): [MIT License](LICENSE)

**Pretrained model weights**: [Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/)
Free to use, share and adapt for non-commercial purposes with attribution.

### Training data sources

| Source | License |
|---|---|
| [Tatoeba DE](https://tatoeba.org) | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) |
| [mC4 DE](https://huggingface.co/datasets/allenai/c4) (Common Crawl) | [ODC-By](https://opendatacommons.org/licenses/by/) |
| Synthetic sentences (generated via [Qwen3](https://huggingface.co/Qwen)) | [Qwen License](https://huggingface.co/Qwen/Qwen3-72B/blob/main/LICENSE) (non-commercial) |

## References

- [FUTO Keyboard source](https://github.com/futo-org/android-keyboard)
- [FUTO LM documentation](https://gitlab.futo.org/keyboard/keyboard-wiki/-/wikis/Keyboard-LM-docs)
- [Issue #1212 — Transformer Models for More Languages](https://github.com/futo-org/android-keyboard/issues/1212)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
