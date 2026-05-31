# FUTO Keyboard — German Transformer Model

Scripts to train a German-language transformer model for [FUTO Keyboard](https://keyboard.futo.org).
FUTO Keyboard ships an English-only next-word-prediction / autocorrect model; this project builds the German equivalent.

**Pretrained models:** [Releases](https://github.com/jblechert/keyboard-lm-de/releases) — drop the `.gguf` into FUTO Keyboard → Settings → Language Models.

---

## Current version: v0.5 — "Drink the Sea" (in training)

> *[Drink the Sea](https://www.youtube.com/watch?v=ezk_dD2Ia-w) — The Glitch Mob*

57M parameter Llama model, significantly expanded training corpus including FineWeb2-HQ (68M sentences) and spoken German from podcasts.
First release expected once training completes at 150k steps.

### Eval results — v0.5 @ 5k steps vs. v0.4 @ 80k steps

Evaluated on 500 freshly generated German sentences (not in either training corpus):

| Metric | v0.4 @ 80k | v0.5 @ 5k | Δ |
|---|---:|---:|---:|
| Avg. Perplexity | 143.6 | **103.9** | −28% |
| Top-1 Accuracy | 24.5% | 24.3% | −0.2% |
| Top-3 Accuracy | 38.9% | **41.9%** | +3.0% |
| Top-5 Accuracy | 47.3% | **49.6%** | +2.3% |
| KSR | 19.8% | **20.0%** | +0.2% |
| Prefix 1 char → Top-1 | 54.6% | **55.2%** | +0.6% |
| Prefix 2 chars → Top-1 | **77.2%** | 77.0% | −0.2% |
| Prefix 3 chars → Top-1 | **88.2%** | 87.4% | −0.8% |

v0.5 @ 5k steps (10% of first epoch) already matches or exceeds v0.4 @ 80k steps on most metrics, with 28% lower perplexity. Full training at 150k steps is expected to improve significantly further.

### Early training loss — v0.5 vs. v0.4

| Step | v0.4 loss | v0.5 loss |
|-----:|----------:|----------:|
| 200 | 7.4876 | 7.1743 |
| 400 | 5.2246 | 4.7564 |
| 600 | 4.0498 | 3.7985 |
| 1000 | 2.6211 | 2.5311 |
| 2000 | 2.1295 | 1.5598 |
| 4600 | — | **1.3309** |
| 20000 | 1.5710 | — |
| 80000 | 1.4327 | — |

Note: loss values are not directly comparable between versions — v0.5 uses a retrained tokenizer (trained on a larger corpus), which produces more coherent token sequences and yields structurally lower cross-entropy loss independent of model quality. Within v0.5, convergence is smooth and consistent with the larger, cleaner training corpus.

### v0.4 — Legacy

v0.4 (50k and 80k checkpoints) is available in the [releases](https://github.com/jblechert/keyboard-lm-de/releases).
It was trained on Tatoeba + mC4 + synthetic data only.
**Note:** v0.4 shows a clear plateau between 65k–80k steps; v0.5 addresses this with a larger and more diverse corpus.

---

## Architecture

| Parameter | Value |
|---|---|
| Architecture | Llama (GGUF via llama.cpp) |
| Parameters | **57M** |
| Layers | 10 × 512 hidden dims, 8 attention heads, 2048 FFN |
| Context | 256 tokens |
| Tokenizer | SentencePiece BPE, `treat_whitespace_as_suffix=true` |

Special autocorrect tokens: `<XBU>`, `<CHAR_A>`…`<CHAR_Z>`, `<XBC>`, `<XEC>`

---

## Training data — v0.5

| Source | Sentences | Weight | License |
|---|---|---|---|
| [Tatoeba DE](https://tatoeba.org) | 770k | 3× | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) |
| [mC4 DE](https://huggingface.co/datasets/allenai/c4) (allenai/c4) | 80M | 1× | [ODC-By](https://opendatacommons.org/licenses/by/) |
| [FineWeb2-HQ DE](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | ~68M | 1× | [ODC-By](https://opendatacommons.org/licenses/by/) |
| Synthetic (Qwen3.6:27b, 27 topics) | ~61k | 3× | generated, non-commercial |
| [Parlamentsrevue](https://parlamentsrevue.de) — Sabrina Gehder | 34k | 2× | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| [Logbuch:Netzpolitik](https://logbuch-netzpolitik.de) — Tim Pritlove & Linus Neumann | 178k | 2× | [CC BY-NC-SA 3.0 DE](https://creativecommons.org/licenses/by-nc-sa/3.0/de/) |
| [Methodisch Inkorrekt!](https://minkorrekt.de) — Nicolas Wöhrl & Reinhard Remfort | 42k | 2× | CC BY-NC-SA 3.0 |
| [Raumzeit](https://raumzeit-podcast.de) — Tim Pritlove | 28k | 2× | [CC BY-NC-SA 3.0 DE](https://creativecommons.org/licenses/by-nc-sa/3.0/de/) |
| [Forschergeist](https://forschergeist.de) — Tim Pritlove | 35k | 2× | [CC BY-NC-SA 3.0 DE](https://creativecommons.org/licenses/by-nc-sa/3.0/de/) |
| [CRE: Technik, Kultur, Gesellschaft](https://cre.fm) — Tim Pritlove | 12k | 2× | [CC BY-NC-SA 3.0 DE](https://creativecommons.org/licenses/by-nc-sa/3.0/de/) |

All Whisper-transcribed podcast data was cleaned: filler words (ähm/äh) replaced, run-ons > 60 words removed, NOTE-block metadata stripped.

---

## Pipeline

### 1. Download training data

```bash
# Tatoeba
.venv_ml/bin/python 07_download_tatoeba.py

# mC4 (80M sentences, ~12h)
.venv_ml/bin/python 09_download_c4_de.py --target 80000000

# FineWeb2-HQ DE (~68M sentences, ~several hours)
.venv_ml/bin/python 24_download_fineweb2_de.py

# Podcasts (all CC BY-NC-SA, Metaebene / Parlamentsrevue)
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

### 4. Train SentencePiece tokenizer

```bash
.venv_ml/bin/python 04_train_tokenizer.py
```

### 5. Train the model

```bash
.venv_ml/bin/python 05_train_model.py --steps 150000 --version v0.5
# ~30 hours on RX 7900 XTX (ROCm)
```

### 6. Convert to GGUF

```bash
.venv_ml/bin/python 06_convert_to_gguf.py
```

---

## Quality metrics (v0.4 @ 80k steps)

| Metric | Value |
|---|---|
| Top-1 Accuracy | 26.8% |
| Top-3 Accuracy | 41.1% |
| Top-5 Accuracy | 48.3% |
| Keystroke Savings Rate (KSR) | 22.3% |
| Prefix 2 chars → Top-1 | 74.2% |
| Prefix 3 chars → Top-3 | 97.2% |

v0.5 targets improvement through 4× larger corpus and spoken German from podcasts.

---

## Requirements

```bash
python -m venv --system-site-packages .venv_ml
.venv_ml/bin/pip install "tokenizers==0.21.0" "transformers>=4.49,<5" datasets gguf sentencepiece
```

- Python 3.10+
- PyTorch with CUDA (NVIDIA) or ROCm (AMD, Linux)
- `bf16=True` requires NVIDIA Ampere (RTX 30xx+) or AMD ROCm — for older cards use `fp16=True`

### Hardware

| GPU | VRAM | 80k steps | 150k steps |
|-----|------|-----------|------------|
| RX 7900 XTX / RTX 4090 | 24 GB | ~16 h | ~30 h |
| RTX 3080 / RTX 4070 | 10–12 GB | ~22 h | ~42 h |
| RTX 3060 12 GB | 12 GB | ~44 h | ~82 h |

---

## License

**Code:** [MIT License](LICENSE)

**Model weights:** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — free for non-commercial use with attribution.

### Training data attribution

| Source | Authors | License |
|---|---|---|
| [Tatoeba DE](https://tatoeba.org) | Tatoeba contributors | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) |
| [mC4 DE](https://huggingface.co/datasets/allenai/c4) | Common Crawl / Allen AI | [ODC-By](https://opendatacommons.org/licenses/by/) |
| [FineWeb2-HQ DE](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | Common Crawl / HuggingFace FineData | [ODC-By](https://opendatacommons.org/licenses/by/) |
| [Parlamentsrevue](https://parlamentsrevue.de) | Sabrina Gehder | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| [Logbuch:Netzpolitik](https://logbuch-netzpolitik.de) | Tim Pritlove & Linus Neumann | [CC BY-NC-SA 3.0 DE](https://creativecommons.org/licenses/by-nc-sa/3.0/de/) |
| [Methodisch Inkorrekt!](https://minkorrekt.de) | Nicolas Wöhrl & Reinhard Remfort | CC BY-NC-SA 3.0 |
| [Raumzeit](https://raumzeit-podcast.de) | Tim Pritlove | [CC BY-NC-SA 3.0 DE](https://creativecommons.org/licenses/by-nc-sa/3.0/de/) |
| [Forschergeist](https://forschergeist.de) | Tim Pritlove | [CC BY-NC-SA 3.0 DE](https://creativecommons.org/licenses/by-nc-sa/3.0/de/) |
| [CRE: Technik, Kultur, Gesellschaft](https://cre.fm) | Tim Pritlove | [CC BY-NC-SA 3.0 DE](https://creativecommons.org/licenses/by-nc-sa/3.0/de/) |
| Synthetic sentences | generated via [Qwen3](https://huggingface.co/Qwen) (Ollama) | non-commercial |

---

## Keyboard Collector App

[`keyboard-collector/`](keyboard-collector/) — Android app to collect correction data for fine-tuning.

Users report misrecognitions (e.g. keyboard predicted "Mahd" instead of "naja") with optional context. Data exports as JSON:

```json
{
  "cases": [
    {
      "recognized": "Mahd",
      "expected": "naja",
      "context": "WhatsApp",
      "ts": "2026-05-31T16:00:00"
    }
  ]
}
```

The exported JSON feeds into a fine-tuning pipeline on top of the v0.5 base model.

**Install:** sideload [`keyboard-collector-debug.apk`](keyboard-collector/keyboard-collector-debug.apk) (debug build, Android 8.0+)

---

## References

- [FUTO Keyboard source](https://github.com/futo-org/android-keyboard)
- [FUTO LM documentation](https://gitlab.futo.org/keyboard/keyboard-wiki/-/wikis/Keyboard-LM-docs)
- [Issue #1212 — Transformer Models for More Languages](https://github.com/futo-org/android-keyboard/issues/1212)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
