# FUTO Keyboard — German Language Model

Scripts to train a German next-word-prediction model for [FUTO Keyboard](https://keyboard.futo.org).
FUTO Keyboard ships an English-only model; this project builds the German equivalent.

**Pretrained models:** [Releases](https://github.com/jblechert/keyboard-lm-de/releases) — drop the `.gguf` into FUTO Keyboard → Settings → Language Models.

---

## Latest release

See [Releases](https://github.com/jblechert/keyboard-lm-de/releases) for the current model and changelog.

### Which quantization should I use?

Start with Q8\_0. It's near-lossless and is the best way to judge quality. Step down if battery drain is noticeable:

| File | Size | Recommendation |
|------|------|----------------|
| Q8\_0 | ~139 MB | Best quality — start here |
| Q6\_K | ~107 MB | Good all-round |
| Q4\_0 | ~77 MB | Older or slower devices |
| Q3\_K\_M | ~67 MB | Very low-end devices |

The keyboard model runs a forward pass on every keypress, so larger files drain battery faster.

---

## Architecture

Llama-based model, GGUF format (llama.cpp compatible).

| Parameter | Value |
|---|---|
| Layers | 12 × 768 hidden, 12 heads, 3072 FFN |
| Parameters | ~136M |
| Context | 256 tokens |
| Tokenizer | SentencePiece BPE (`treat_whitespace_as_suffix=true`) |

Special autocorrect tokens: `<XBU>`, `<CHAR_A>`…`<CHAR_Z>`, `<XBC>`, `<XEC>`

---

## Training data

| Source | License |
|---|---|
| [Tatoeba DE](https://tatoeba.org) | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) |
| [mC4 DE](https://huggingface.co/datasets/allenai/c4) | [ODC-By](https://opendatacommons.org/licenses/by/) |
| [FineWeb2-HQ DE](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | [ODC-By](https://opendatacommons.org/licenses/by/) |
| Synthetic (Qwen3.6:27b via Ollama, 28 topics) | generated, non-commercial |
| [Parlamentsrevue](https://parlamentsrevue.de) — Sabrina Gehder | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| [Logbuch:Netzpolitik](https://logbuch-netzpolitik.de) — Tim Pritlove & Linus Neumann | [CC BY-NC-SA 3.0 DE](https://creativecommons.org/licenses/by-nc-sa/3.0/de/) |
| [Methodisch Inkorrekt!](https://minkorrekt.de) — Nicolas Wöhrl & Reinhard Remfort | CC BY-NC-SA 3.0 |
| [Raumzeit](https://raumzeit-podcast.de) — Tim Pritlove | [CC BY-NC-SA 3.0 DE](https://creativecommons.org/licenses/by-nc-sa/3.0/de/) |
| [Forschergeist](https://forschergeist.de) — Tim Pritlove | [CC BY-NC-SA 3.0 DE](https://creativecommons.org/licenses/by-nc-sa/3.0/de/) |
| [CRE: Technik, Kultur, Gesellschaft](https://cre.fm) — Tim Pritlove | [CC BY-NC-SA 3.0 DE](https://creativecommons.org/licenses/by-nc-sa/3.0/de/) |

---

## Pipeline

### 1. Download training data

```bash
.venv_ml/bin/python 07_download_tatoeba.py
.venv_ml/bin/python 09_download_c4_de.py --target 80000000
.venv_ml/bin/python 24_download_fineweb2_de.py
.venv_ml/bin/python 15_download_parlamentsrevue.py
.venv_ml/bin/python 16_download_lnp.py
.venv_ml/bin/python 17_download_minkorrekt.py
.venv_ml/bin/python 18_download_raumzeit.py
.venv_ml/bin/python 19_download_forschergeist.py
.venv_ml/bin/python 20_download_cre.py
```

### 2. Clean training data

```bash
.venv_ml/bin/python 10_clean_training_data.py
```

### 3. Generate synthetic sentences (requires Ollama + qwen3.6:27b)

```bash
.venv_ml/bin/python 12_generate_synthetic_vocab.py --per-topic 2000
```

### 4. Train tokenizer

```bash
.venv_ml/bin/python 04_train_tokenizer.py
```

### 5. Train model

```bash
.venv_ml/bin/python 05_train_model.py --steps 200000
```

### 6. Convert to GGUF

```bash
.venv_ml/bin/python 06_convert_to_gguf.py
```

---

## Requirements

```bash
python -m venv --system-site-packages .venv_ml
.venv_ml/bin/pip install "tokenizers==0.21.0" "transformers>=4.49,<5" datasets gguf sentencepiece
```

- Python 3.10+
- PyTorch with CUDA (NVIDIA) or ROCm (AMD, Linux)
- GPU with at least 12 GB VRAM recommended

---

## Keyboard Collector App

[`keyboard-collector/`](keyboard-collector/) — Android app to collect correction data for fine-tuning.

Reports misrecognitions (e.g. keyboard predicted "Mahd" instead of "naja") with optional context. Data exports as JSON for fine-tuning.

**Install:** sideload [`keyboard-collector-debug.apk`](keyboard-collector/keyboard-collector-debug.apk) (Android 8.0+)

---

## License

**Code:** [MIT License](LICENSE)

**Model weights:** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — free for non-commercial use with attribution.

---

## References

- [FUTO Keyboard](https://keyboard.futo.org) / [Source](https://github.com/futo-org/android-keyboard)
- [FUTO LM documentation](https://gitlab.futo.org/keyboard/keyboard-wiki/-/wikis/Keyboard-LM-docs)
- [Issue #1212 — Transformer Models for More Languages](https://github.com/futo-org/android-keyboard/issues/1212)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
