# v0.4 Training Plan

## Architecture decision: Base + LoRA

Previous versions (v0.2, v0.3) mixed all data with weights in one training pass.
Problem: synthetic data (~2k sentences) got repeated ~130k times while c4 barely
completed one pass. From v0.4 onward we split into two phases.

---

## Phase 1 — Base model (c4 only)

**Goal:** Solid German language model, no keyboard-specific biases.

| Parameter | Value |
|---|---|
| Data | `data/c4_de.txt` only (20.4M sentences, cleaned) |
| Steps | 200k |
| Weights | c4 1× (no mixing) |
| Synthetic weight | 3× (down from 6×) — but synthetic goes into LoRA, not base |
| Est. time | ~55h on RX 7900 XTX |

```bash
.venv_ml/bin/python 05_train_model.py --steps 200000
# (after removing tatoeba/synthetic from mixed_generator, c4-only mode)
```

Milestone snapshots every 10k steps for perplexity tracking.

---

## Phase 2 — LoRA: Keyboard style + Autocorrect

**Goal:** Steer toward everyday German keyboard usage; teach XBU/CHAR autocorrect.

| Parameter | Value |
|---|---|
| Base | Phase 1 checkpoint (200k steps) |
| Data | Tatoeba (770k) + synthetic_*.txt (target: 20k–40k sentences) |
| XBU augmentation | `18_generate_xbu_data.py --copies 3` → 4× lines, ~30% words XBU-converted |
| LoRA rank | 32 |
| Steps | ~5k–10k (fast, dataset is small) |
| Synthetic weight | 3× relative to Tatoeba |

```bash
# 1. Generate XBU training data
.venv_ml/bin/python 18_generate_xbu_data.py --copies 3

# 2. Fine-tune
.venv_ml/bin/python 14_finetune_lora.py --data data/xbu_lora.txt

# 3. Merge + convert
.venv_ml/bin/python 15_merge_lora.py
.venv_ml/bin/python 06_convert_to_gguf.py
```

Ticks: XBU/CHAR autocorrect fine-tuning ✓

---

## Phase 3 — Topic LoRA adapters (pluggable)

**Goal:** Domain-specific vocabulary and phrasing on top of the base model.
Each topic is a separate LoRA adapter, merged individually or kept separate.

| Topic | Data source | Status |
|---|---|---|
| Computing / IT | Synthetic (Qwen3) + filtered c4 tech sentences | planned |
| Medicine | Synthetic (Qwen3) + filtered c4 medical sentences | planned |
| Legal / Behörden | Synthetic (Qwen3) | planned |
| Cooking | Synthetic (Qwen3) | planned |

**Approach per topic:**
1. Generate ~5k–10k topic-specific synthetic sentences via `08_generate_synthetic.py`
   with a domain-specific system prompt
2. Optionally: filter matching sentences from c4 (keyword-based)
3. LoRA fine-tune on top of Phase 1 base (not Phase 2 — keep keyboard style separate)
4. Ship as optional `.gguf` sidecar or merged variant

Topic adapters are additive — a user can load the base keyboard model +
the computing adapter for tech-heavy typing contexts.

---

## Data status

| File | Lines | Status |
|---|---|---|
| `data/c4_de.txt` | 20.4M | ✅ cleaned (2026-05-28) |
| `data/tatoeba_de.txt` | 770k | ✅ ready |
| `data/synthetic_de.txt` | growing (4k+) | 🔄 generating overnight |
| `data/xbu_lora.txt` | — | ⬜ generate after synthetic done |

---

## Immediate next steps

1. ⬜ Let synthetic generation run overnight → target 20k+ sentences
2. ⬜ Review synthetic quality sample, stop when satisfied
3. ⬜ `18_generate_xbu_data.py` → `data/xbu_lora.txt`
4. ⬜ Adapt `05_train_model.py` for c4-only base training mode
5. ⬜ Start Phase 1 base training
6. ⬜ While base trains: design topic synthetic prompts for Phase 3
