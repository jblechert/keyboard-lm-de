#!/usr/bin/env python3
"""
Modellanalyse: Perplexity, Vorhersagehäufigkeit und Top-K-Accuracy

1. Perplexity-Analyse
   Berechnet die Perplexity für eine Stichprobe aus den Trainingsdaten.
   Sätze mit sehr hoher Perplexity sind Kandidaten für die BANNED-Liste
   in 10_clean_training_data.py — das Modell hat sie nicht gut gelernt,
   oft weil sie Rauschen, Spam oder Fremdsprachanteile enthalten.

2. Vorhersagehäufigkeits-Analyse
   Generiert für viele zufällige Kontexte die Top-K-Vorhersagen und
   zählt, welche Wörter wie oft vorhergesagt werden. Wörter die deutlich
   häufiger vorhergesagt werden als sie im Korpus vorkommen, sind
   potentielle Überrepräsentierungsprobleme (wie "Mahd").

3. Top-K-Accuracy
   Misst wie oft das richtige nächste Wort in den Top-1/3/5-Vorschlägen
   landet — direkte Entsprechung zur gefühlten Qualität auf dem Keyboard.
   Zusätzlich: Keystroke Savings Rate (KSR), der Industriestandard für
   Keyboard-LM-Qualität.

Usage:
  .venv_ml/bin/python 11_analyze_model.py [--samples 2000] [--contexts 500]
  .venv_ml/bin/python 11_analyze_model.py --skip-perplexity --skip-frequency
"""

import argparse
import math
import random
from collections import Counter
from pathlib import Path

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

MODEL_DIR  = Path("data/model_hf")
SP_MODEL   = Path("data/tokenizer/de_keyboard.model")

SOURCES = [
    Path("data/tatoeba_de.txt"),
    Path("data/c4_de.txt"),
    *sorted(Path("data").glob("synthetic_*.txt")),
]

CONTEXT_LEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # überschreibbar via --device


def load_model_and_tokenizer():
    print("Lade Tokenizer und Modell …")
    tok = LlamaTokenizer(vocab_file=str(SP_MODEL), legacy=False)
    tok.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>",
                            "unk_token": "<unk>", "pad_token": "<unk>"})
    model = LlamaForCausalLM.from_pretrained(str(MODEL_DIR), torch_dtype=torch.float32)
    model.eval()
    model.to(DEVICE)
    return model, tok


def load_sentences(n: int) -> list[str]:
    """Zieht n zufällige Sätze aus allen Quellen (gewichtet nach Dateigröße)."""
    all_lines = []
    for src in SOURCES:
        if src.exists():
            all_lines.extend(src.read_text(encoding="utf-8").splitlines())
    random.shuffle(all_lines)
    return [l.strip() for l in all_lines if len(l.strip()) > 20][:n]


# ── 1. Perplexity ─────────────────────────────────────────────────────────────

def sentence_perplexity(model, tok, sentence: str) -> float:
    ids = tok.encode(sentence, add_special_tokens=True)
    if len(ids) < 2:
        return float("nan")
    ids = ids[:CONTEXT_LEN]
    input_ids = torch.tensor([ids], device=DEVICE)
    with torch.no_grad():
        loss = model(input_ids=input_ids, labels=input_ids).loss
    return math.exp(loss.item())


def run_perplexity(model, tok, sentences: list[str], high_threshold: float,
                   export_file: str | None = None):
    print(f"\n── Perplexity-Analyse ({len(sentences)} Sätze) ──")
    results = []
    for i, s in enumerate(sentences):
        ppl = sentence_perplexity(model, tok, s)
        results.append((ppl, s))
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(sentences)} …")

    results.sort(reverse=True)
    finite = [(p, s) for p, s in results if math.isfinite(p)]

    avg_ppl = sum(p for p, _ in finite) / len(finite) if finite else 0
    high = [(p, s) for p, s in finite if p > high_threshold]
    print(f"\nDurchschnittliche Perplexity: {avg_ppl:.1f}")
    print(f"Sätze über {high_threshold:.0f} (Kandidaten zum Entfernen): {len(high)}")

    print(f"\nTop-30 Sätze mit höchster Perplexity:")
    for ppl, s in finite[:30]:
        print(f"  {ppl:8.1f}  {s[:100]}")

    if export_file:
        out = Path(export_file)
        out.write_text("\n".join(s for _, s in high) + "\n", encoding="utf-8")
        print(f"\n{len(high)} Sätze → {out}")

    return avg_ppl, finite


# ── 2. Vorhersagehäufigkeit ───────────────────────────────────────────────────

def get_top_predictions(model, tok, context: str, top_k: int = 10) -> list[str]:
    """Gibt die top_k vorhergesagten nächsten vollständigen Wörter zurück.

    Im Suffix-Space-Tokenizer enden vollständige Wörter mit ▁ (U+2581).
    Nur solche Token werden gezählt — Subword-Fragmente wie 'st', 'en', 'ge'
    werden ignoriert.
    """
    prompt = context.strip() + " "
    ids = tok.encode(prompt, add_special_tokens=True)[-CONTEXT_LEN:]
    input_ids = torch.tensor([ids], device=DEVICE)

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0, -1]

    top_ids = torch.topk(logits, top_k * 10).indices.tolist()
    words = []
    for tid in top_ids:
        piece = tok.convert_ids_to_tokens(tid) or ""
        # Vollständiges Wort = Token endet mit ▁ (Suffix-Space-Konvention)
        if piece.endswith("▁"):
            word = piece[:-1]  # ▁ entfernen
            if word and word.replace("-", "").isalpha() and len(word) > 1:
                words.append(word)
        if len(words) >= top_k:
            break
    return words


def build_corpus_freq(sentences: list[str]) -> Counter:
    freq: Counter = Counter()
    for s in sentences:
        for w in s.split():
            w = w.strip(".,!?;:\"'()[]").lower()
            if w.isalpha():
                freq[w] += 1
    return freq


def run_frequency(model, tok, sentences: list[str], n_contexts: int, top_k: int):
    print(f"\n── Vorhersagehäufigkeits-Analyse ({n_contexts} Kontexte, top-{top_k}) ──")

    # Zufällige 2-4-Wort-Kontexte aus echten Sätzen
    contexts = []
    for s in random.sample(sentences, min(n_contexts * 2, len(sentences))):
        words = s.split()
        if len(words) >= 4:
            cut = random.randint(2, min(5, len(words) - 1))
            contexts.append(" ".join(words[:cut]))
    contexts = contexts[:n_contexts]

    prediction_freq: Counter = Counter()
    for i, ctx in enumerate(contexts):
        preds = get_top_predictions(model, tok, ctx, top_k)
        prediction_freq.update(preds)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_contexts} …")

    corpus_freq = build_corpus_freq(sentences)
    total_corpus = sum(corpus_freq.values())
    total_preds  = sum(prediction_freq.values())

    print(f"\nTop-50 am häufigsten vorhergesagte Wörter")
    print(f"{'Wort':<20} {'Vorhersagen':>12} {'Korpus-Anteil':>14} {'Ratio':>8}")
    print("─" * 58)

    for word, pred_count in prediction_freq.most_common(50):
        pred_share   = pred_count / total_preds * 100
        corpus_share = corpus_freq.get(word.lower(), 0) / total_corpus * 100
        ratio = pred_share / corpus_share if corpus_share > 0 else float("inf")
        flag = "  ← ?" if ratio > 20 and corpus_share < 0.05 else ""
        print(f"  {word:<18} {pred_count:>10}  {corpus_share:>12.3f}%  {ratio:>7.1f}×{flag}")

    return prediction_freq, corpus_freq


# ── 3. Top-K-Accuracy & Keystroke Savings Rate ───────────────────────────────

def run_accuracy(model, tok, sentences: list[str], n_samples: int):
    """
    Berechnet Top-1/3/5-Accuracy und Keystroke Savings Rate (KSR).

    Für jede Position nach einem vollständigen Wort (Token endet mit ▁) wird
    geprüft, ob das nächste vollständige Wort in den Top-K Vorhersagen landet.

    KSR: Anteil der Tastenanschläge die durch korrekte Top-1-Vorhersage gespart
    werden. Formel: KSR = Σ(gesparte_zeichen) / Σ(zeichen_ohne_vorhersage)
    Entspricht dem Industriestandard für Keyboard-LM-Evaluation.
    """
    TOP_K_LIST = [1, 3, 5]
    SCAN = 50   # wie viele Kandidaten-Token pro Position scannen

    hits   = {k: 0 for k in TOP_K_LIST}
    total  = 0
    saved_keystrokes   = 0
    total_keystrokes   = 0

    sample = random.sample(sentences, min(n_samples, len(sentences)))

    print(f"\n── Top-K-Accuracy ({len(sample)} Sätze) ──")

    for idx, sent in enumerate(sample):
        ids = tok.encode(sent, add_special_tokens=True)
        if len(ids) < 3:
            continue
        ids = ids[:CONTEXT_LEN]

        for pos in range(1, len(ids) - 1):
            prev_piece = tok.convert_ids_to_tokens(ids[pos - 1]) or ""
            true_piece = tok.convert_ids_to_tokens(ids[pos]) or ""

            # Nur auswerten wenn vorheriges Token ein vollständiges Wort ist
            # und das wahre nächste Token ebenfalls ein vollständiges Wort
            if not prev_piece.endswith("▁") or not true_piece.endswith("▁"):
                continue

            true_word = true_piece[:-1].lower()
            if not true_word or not true_word.replace("-", "").isalpha() or len(true_word) < 2:
                continue

            context_ids = ids[:pos]
            input_ids   = torch.tensor([context_ids], device=DEVICE)
            with torch.no_grad():
                logits = model(input_ids=input_ids).logits[0, -1]

            # Top-5 vollständige Wörter aus den besten SCAN Kandidaten
            top_ids = torch.topk(logits, SCAN * len(TOP_K_LIST)).indices.tolist()
            predicted: list[str] = []
            for tid in top_ids:
                piece = tok.convert_ids_to_tokens(tid) or ""
                if piece.endswith("▁"):
                    w = piece[:-1].lower()
                    if w and w.replace("-", "").isalpha() and len(w) > 1:
                        predicted.append(w)
                if len(predicted) >= max(TOP_K_LIST):
                    break

            total += 1
            total_keystrokes += len(true_word)
            for k in TOP_K_LIST:
                if true_word in predicted[:k]:
                    hits[k] += 1
            if true_word == predicted[0] if predicted else "":
                saved_keystrokes += len(true_word)

        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(sample)} Sätze …")

    if total == 0:
        print("  Keine auswertbaren Positionen gefunden.")
        return

    print(f"\n  Auswertbare Wortpositionen: {total:,}")
    print()
    for k in TOP_K_LIST:
        acc = hits[k] / total * 100
        print(f"  Top-{k} Accuracy:  {acc:5.1f}%  ({hits[k]:,}/{total:,})")

    ksr = saved_keystrokes / total_keystrokes * 100 if total_keystrokes else 0
    avg_word_len = total_keystrokes / total if total else 0
    print(f"\n  Keystroke Savings Rate (KSR):  {ksr:.1f}%")
    print(f"  (Ø Wortlänge: {avg_word_len:.1f} Zeichen, Top-1 korrekt → alles gespart)")

    return hits, total, ksr


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples",      type=int, default=2000,
                        help="Sätze für Perplexity-Analyse (default: 2000)")
    parser.add_argument("--contexts",     type=int, default=500,
                        help="Kontexte für Häufigkeitsanalyse (default: 500)")
    parser.add_argument("--top-k",        type=int, default=10,
                        help="Top-K Vorhersagen pro Kontext (default: 10)")
    parser.add_argument("--ppl-threshold",type=float, default=200.0,
                        help="Perplexity-Schwelle für Ausreißer (default: 200)")
    parser.add_argument("--device", default=None,
                        help="Gerät: cuda, cpu (default: auto)")
    parser.add_argument("--model-dir", default=str(MODEL_DIR),
                        help=f"Modellverzeichnis (default: {MODEL_DIR})")
    parser.add_argument("--accuracy-samples", type=int, default=500,
                        help="Sätze für Top-K-Accuracy (default: 500)")
    parser.add_argument("--skip-perplexity",  action="store_true")
    parser.add_argument("--skip-frequency",   action="store_true")
    parser.add_argument("--skip-accuracy",    action="store_true")
    parser.add_argument("--export-high-ppl",  metavar="FILE", default=None,
                        help="Sätze über --ppl-threshold in diese Datei schreiben "
                             "(kompatibel mit 10_clean_training_data.py --high-ppl)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    global DEVICE, MODEL_DIR
    if args.device:
        DEVICE = args.device
    MODEL_DIR = Path(args.model_dir)
    print(f"Modell: {MODEL_DIR}  |  Gerät: {DEVICE}")

    random.seed(args.seed)

    if not MODEL_DIR.exists():
        print(f"Fehler: Kein Modell in {MODEL_DIR}", flush=True)
        return

    model, tok = load_model_and_tokenizer()
    sentences = load_sentences(max(args.samples, args.contexts * 3, args.accuracy_samples))
    print(f"{len(sentences)} Sätze geladen.")

    if not args.skip_perplexity:
        run_perplexity(model, tok, sentences[:args.samples], args.ppl_threshold,
                       export_file=args.export_high_ppl)

    if not args.skip_frequency:
        run_frequency(model, tok, sentences, args.contexts, args.top_k)

    if not args.skip_accuracy:
        run_accuracy(model, tok, sentences, args.accuracy_samples)


if __name__ == "__main__":
    main()
