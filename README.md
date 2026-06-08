# FUTO Keyboard — German Transformer Model

Scripts to train a German-language transformer model for [FUTO Keyboard](https://keyboard.futo.org).
FUTO Keyboard ships an English-only next-word-prediction / autocorrect model; this project builds the German equivalent.

**Pretrained models:** [Releases](https://github.com/jblechert/keyboard-lm-de/releases) — drop the `.gguf` into FUTO Keyboard → Settings → Language Models.

---

## Current version: v0.5.1 — "To the max" (in training)

> *Training in progress — 200k steps, ETA ~55h on RX 7900 XTX*

136M parameter Llama model — 2.4× bigger than v0.5. Larger architecture (768 hidden, 12 layers, 12 heads, 3072 FFN), cleaner training data, new colloquial German topic.
A Q4 quantization of this model is expected to outperform v0.5 F16.

### What changed vs. v0.5

**Architecture:** 512 → 768 hidden dims, 10 → 12 layers, 8 → 12 attention heads, 2048 → 3072 FFN (~57M → ~136M parameters)

**Data quality:** c4 and FineWeb2 re-cleaned with sharpened filters — 1.74M sentences removed (Weeze, Schwedt, Meckenheim, Fine, Banater, Fünen, MDK, MdL, NDP, LSK, Fibu, SSe, Leu, Chet, j-ja, Fonem, Mediendom, Aalenerin and more)

**Synthetic data:** New topic *Umgangssprache* — colloquial contractions (was'n, geht's, willste, haste, biste, kannste …) that appear on every German keyboard but were underrepresented

**Tokenizer:** Retrained on the full cleaned corpus including FineWeb2 and the new synthetic topic

### Eval results — v0.5 @ 105k vs. v0.5 @ 150k (HF checkpoint, no quantization)

| Metric | v0.5 @ 105k | v0.5 @ 150k |
|---|---:|---:|
| Loss | 0.8391 | **0.7900** |
| Top-1 Accuracy | 67.4% | **67.5%** |
| Top-3 Accuracy | **82.9%** | 82.8% |
| Top-5 Accuracy | 89.1% | **89.3%** |
| KSR | 66.8% | **67.0%** |

v0.5 did not plateau at 105k as initially believed — it continued improving until step 143k (loss minimum 0.790), then slightly rose as the cosine LR schedule ended. The accuracy plateau despite continued loss improvement suggests the model was saturating its capacity — motivation for the larger v0.5.1 architecture.

## Previous version: v0.5 — "Drink the Sea"

> *[Drink the Sea](https://www.youtube.com/watch?v=ezk_dD2Ia-w) — The Glitch Mob*

57M parameter Llama model, significantly expanded training corpus including FineWeb2-HQ (68M sentences) and spoken German from podcasts.

### Eval results — v0.5 @ 5k steps vs. v0.4 @ 80k steps

Evaluated on 500 freshly generated German sentences (not in either training corpus):

| Metric | v0.4 @ 80k | v0.5 @ 75k | v0.5 @ 100k |
|---|---:|---:|---:|
| Top-1 Accuracy | 24.5% | 26.6% | **26.7%** |
| Top-3 Accuracy | 38.9% | 43.9% | **44.3%** |
| Top-5 Accuracy | 47.3% | 52.3% | **53.6%** |
| KSR | 19.8% | 22.5% | **22.7%** |
| Prefix 2 chars → Top-1 | 77.2% | — | **78.7%** |
| Prefix 3 chars → Top-1 | 88.2% | — | **88.4%** |

v0.5 @ 100k ≈ 2 full epochs. Training ran to 150k steps (loss minimum 0.790 @ 143k).

### Quantization comparison — v0.5 @ 5k steps

Evaluated on 500 freshly generated German sentences:

| Metric | F32 (HF) | Q8_0 | Q4_0 |
|---|---:|---:|---:|
| Top-1 Accuracy | **24.6%** | 23.5% | 23.1% |
| Top-3 Accuracy | **41.3%** | 40.0% | 40.1% |
| Top-5 Accuracy | **48.6%** | 48.7% | 48.6% |
| KSR | **20.6%** | 19.5% | 18.9% |
| Prefix 2 chars → Top-1 | **77.5%** | 77.4% | 77.0% |
| Prefix 3 chars → Top-1 | **86.9%** | 87.2% | 86.9% |
| Size | 110 MB | 59 MB | 34 MB |

Q4_0 loses ~1.5% Top-1 and ~1.7% KSR vs F32 — acceptable for a 3× size reduction. Q8_0 is nearly lossless.

### Early training loss — v0.5 vs. v0.4

| Step | v0.4 loss | v0.5 loss |
|-----:|----------:|----------:|
| 1,000 | 2.6211 | 2.5311 |
| 2,000 | 2.1295 | 1.5598 |
| 5,000 | 1.8749 | 1.3110 |
| 10,000 | 1.7216 | 1.1535 |
| 15,000 | 1.6165 | 1.0600 |
| 20,000 | 1.5710 | 1.0324 |
| 25,000 | 1.5668 | 1.0374 |
| 30,000 | 1.5202 | 1.0028 |
| 35,000 | 1.5258 | 0.9866 |
| 50,000 | 1.5001 | 0.9364 |
| 65,000 | 1.4639 | 0.9142 |
| 75,000 | 1.4527 | 0.9142 |
| 80,000 | 1.4327 | 0.8856 |
| 100,000 | — | 0.8542 |
| 135,000 | — | **0.8017** |

**What is loss?** At each position in a sentence the model predicts the next word. Loss (cross-entropy) measures how wrong those predictions are on average — specifically, it's the negative log-probability the model assigns to the correct next word: `loss = -ln(p_correct)`. Lower is better.

- **Loss 9.6** — random guessing across a 15,000-word vocabulary (the starting point)
- **Loss 2.0** — the model has learned basic German grammar and common words
- **Loss 1.4** — v0.4's final level: reasonable predictions, noticeable plateau
- **Loss 0.8** — v0.5's level: the model is confident and correct on most positions
- **Loss 0.0** — would mean perfect memorization (undesirable; never reached in practice)

**Training perplexity vs. external perplexity** — these look very different and that's expected. At 144k steps v0.5 has training perplexity **2.2** but external eval perplexity **~286**. This is not overfitting. Training perplexity is computed on the same data distribution the model was trained on, with tokenizer-optimized sequences — naturally easy for the model. External perplexity uses freshly generated sentences with rare compound words and unusual phrasings that push individual token predictions to high uncertainty. What matters is that external perplexity *falls* over training (392 at 50k → 303 at 100k → 286 at 135k), proving the model is genuinely generalizing better, not just memorizing.

**Why Euler's number?** Loss uses the natural logarithm (ln, base *e* ≈ 2.718). This means the inverse operation is *e^loss*, which gives *perplexity* — the effective number of equally likely candidates the model is choosing from at each word position. At loss 0.8: *e^0.8 ≈ 2.2 candidates*. At loss 1.4: *e^1.4 ≈ 4.1 candidates*. At loss 9.6 (random): *e^9.6 ≈ 14,765* — roughly the full vocabulary size. Euler's number appears here because PyTorch computes cross-entropy using natural logarithm, which has the simplest mathematical properties for gradient-based optimization (its derivative is 1/x).

Note: loss values are not directly comparable between model versions that use different tokenizers — v0.5 uses a retrained tokenizer which changes the token boundaries. Use the accuracy/KSR metrics in the eval table for fair cross-version comparison.

**Should you try to lower perplexity further?** Perplexity is a proxy, not the real goal. What actually matters for a keyboard LM is **KSR and Top-K accuracy** — the metrics that feel good to users.

Corpus cleaning lowers external perplexity indirectly (the model stops wasting capacity on spam, broken sentences, and foreign text), but perplexity has a fundamental limitation: it measures whether the model predicts *statistically likely* next words, not whether it predicts *what the user actually wants*. A model can have low perplexity and still suggest "Mahd" when the user meant "naja" — because "Mahd" is a real German word that appears in the training data.

The most effective levers for improving prediction quality, roughly in order of impact:

| Lever | Effect |
|---|---|
| Corpus cleaning | Removes noise → model capacity used for real language |
| Targeted synthetic data | Covers vocabulary gaps (e.g. "zubuchen", domain words) |
| More training / better LR schedule | Slow and steady improvement across all metrics |
| User correction fine-tuning | Directly fixes specific wrong predictions that bother users |

The last lever — collecting real misrecognitions via the [Keyboard Collector app](#keyboard-collector-app) and fine-tuning on them — is the most targeted. It doesn't improve general language knowledge; it fixes the exact cases that frustrate actual users.

**On corpus cleaning: less is more.** It is tempting to ban many words and patterns aggressively. Resist this. Every sentence you remove also removes context the model needs to understand when *not* to predict a word. Ban a word only when you are certain it is wrong in all contexts — not just in the context where you noticed it. A targeted ban list of 20–30 clearly wrong patterns is more effective than 500 aggressive rules that remove legitimate training signal. Observe model outputs for a while, collect specific failures, and ban only what you can confidently call noise.

### v0.4 — Legacy

v0.4 (50k and 80k checkpoints) is available in the [releases](https://github.com/jblechert/keyboard-lm-de/releases).
It was trained on Tatoeba + mC4 + synthetic data only.
**Note:** v0.4 shows a clear plateau between 65k–80k steps; v0.5 addresses this with a larger and more diverse corpus.

---

## Architecture

| Parameter | v0.5 | v0.5.1 |
|---|---|---|
| Architecture | Llama (GGUF via llama.cpp) | Llama (GGUF via llama.cpp) |
| Parameters | 57M | **136M** |
| Layers | 10 × 512 hidden, 8 heads, 2048 FFN | **12 × 768 hidden, 12 heads, 3072 FFN** |
| Context | 256 tokens | 256 tokens |
| Tokenizer | SentencePiece BPE, `treat_whitespace_as_suffix=true` | SentencePiece BPE, `treat_whitespace_as_suffix=true` |

Special autocorrect tokens: `<XBU>`, `<CHAR_A>`…`<CHAR_Z>`, `<XBC>`, `<XEC>`

---

## Which quantization should I use?

**Start with Q8_0.** It offers near-lossless quality and is the best way to judge if the model works well for your use case. If it feels sluggish or your battery drains noticeably faster, step down:

| File | Size | Battery impact | Recommendation |
|------|------|---------------|----------------|
| Q8_0 | 59 MB | highest (~2×) | Try this first |
| Q6_K | 46 MB | medium-high | Good all-round |
| Q4_0 | 34 MB | medium | Older/slower devices |
| Q3_K_M | 30 MB | lowest | Very old/low-end devices |

Battery impact is roughly proportional to file size — the keyboard model runs a forward pass on every keypress, so Q8 reads about twice as much data from RAM as Q3_K_M per prediction. On a phone typing 10,000 characters a day this adds up. If you notice faster battery drain, switch to Q4_0 or Q3_K_M.

**Note on quantization quality:** Precise per-quantization accuracy differences could not be reliably measured with the current evaluation tooling due to a tokenizer handling mismatch between the GGUF runtime and the HuggingFace evaluation pipeline. In practice, Q8_0 is near-lossless, Q4_0 causes a small but noticeable quality reduction, and Q3_K_M is a further step down — the best way to judge is to try Q8_0 first and step down only if needed.

---

## Training data — v0.5.1

| Source | Sentences | Weight | License |
|---|---|---|---|
| [Tatoeba DE](https://tatoeba.org) | 770k | 3× | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) |
| [mC4 DE](https://huggingface.co/datasets/allenai/c4) (allenai/c4) | ~73M ¹ | 1× | [ODC-By](https://opendatacommons.org/licenses/by/) |
| [FineWeb2-HQ DE](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | ~67M ¹ | 1× | [ODC-By](https://opendatacommons.org/licenses/by/) |
| Synthetic (Qwen3.6:27b, 28 topics) | ~66k | 3× | generated, non-commercial |
| [Parlamentsrevue](https://parlamentsrevue.de) — Sabrina Gehder | 34k | 2× | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| [Logbuch:Netzpolitik](https://logbuch-netzpolitik.de) — Tim Pritlove & Linus Neumann | 178k | 2× | [CC BY-NC-SA 3.0 DE](https://creativecommons.org/licenses/by-nc-sa/3.0/de/) |
| [Methodisch Inkorrekt!](https://minkorrekt.de) — Nicolas Wöhrl & Reinhard Remfort | 42k | 2× | CC BY-NC-SA 3.0 |
| [Raumzeit](https://raumzeit-podcast.de) — Tim Pritlove | 28k | 2× | [CC BY-NC-SA 3.0 DE](https://creativecommons.org/licenses/by-nc-sa/3.0/de/) |
| [Forschergeist](https://forschergeist.de) — Tim Pritlove | 35k | 2× | [CC BY-NC-SA 3.0 DE](https://creativecommons.org/licenses/by-nc-sa/3.0/de/) |
| [CRE: Technik, Kultur, Gesellschaft](https://cre.fm) — Tim Pritlove | 12k | 2× | [CC BY-NC-SA 3.0 DE](https://creativecommons.org/licenses/by-nc-sa/3.0/de/) |

¹ After cleaning: 938k removed from c4, 799k removed from FineWeb2 (web spam, broken encoding, foreign language, hyperlocal place names, obscure abbreviations)

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
.venv_ml/bin/python 05_train_model.py --steps 200000 --version v0.5.1
# ~55 hours on RX 7900 XTX (ROCm)
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
