#!/usr/bin/env python3
"""
check_backtranslation.py  --  Preview BT output for ES / HI
============================================================
Loads English ParaDeHate pairs, translates both sides with NLLB-200,
shows results side-by-side, and flags KEEP vs DISCARD via the similarity
guard: sim(tgt_toxic, tgt_neutral) > max_sim → collapsed rewrite → DISCARD.

No toxicity filter is applied (NLLB softens profanity but pairs are kept).

Usage
-----
    python check_backtranslation.py                        # 50 samples, ES + HI
    python check_backtranslation.py --lang es              # Spanish only
    python check_backtranslation.py --lang hi --n 20       # 20 Hindi samples
    python check_backtranslation.py --no-filter-score      # skip similarity scoring
    python check_backtranslation.py --save-pairs output/bt_pairs/   # save to txt files

Requirements
------------
    pip install transformers datasets torch sentence-transformers
"""

import argparse, textwrap, sys
from pathlib import Path

# ── Colour helpers ─────────────────────────────────────────────────────────────
tty = sys.stdout.isatty()
RED    = "\033[91m" if tty else ""
GREEN  = "\033[92m" if tty else ""
CYAN   = "\033[96m" if tty else ""
YELLOW = "\033[93m" if tty else ""
BOLD   = "\033[1m"  if tty else ""
DIM    = "\033[2m"  if tty else ""
RESET  = "\033[0m"  if tty else ""
def col(t, c): return f"{c}{t}{RESET}"
def wrap(t, w=85): return textwrap.fill(t, width=w)

# ── NLLB language codes ────────────────────────────────────────────────────────
NLLB_CODES = {"es": "spa_Latn", "hi": "hin_Deva", "en": "eng_Latn"}

NLLB_MODEL = "facebook/nllb-200-distilled-600M"
DATA_CACHE  = Path(__file__).parent / "output" / "poly_detox" / "data_cache"


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load source EN pairs from ParaDeHate
# ─────────────────────────────────────────────────────────────────────────────

def load_paradehate(n: int, seed: int = 42):
    from datasets import load_dataset
    import random
    random.seed(seed)
    print(col("Loading ParaDeHate (ScaDSAI/ParaDeHate) ...", DIM))

    # Try loading — dataset may have no named split or use "train"
    raw = load_dataset("ScaDSAI/ParaDeHate", cache_dir=str(DATA_CACHE))
    print(col(f"  Available splits: {list(raw.keys())}", DIM))

    # Pick first available split
    split_name = "train" if "train" in raw else list(raw.keys())[0]
    ds = raw[split_name]
    print(col(f"  Using split '{split_name}' — {len(ds)} rows", DIM))
    print(col(f"  Columns: {ds.column_names}", DIM))

    # Detect toxic/neutral column names flexibly
    cols = ds.column_names
    tox_col = next((c for c in cols if "original" in c.lower() or "toxic" in c.lower()), cols[0])
    neu_col = next((c for c in cols if "converted" in c.lower() or "neutral" in c.lower()), cols[1])
    print(col(f"  Using toxic='{tox_col}', neutral='{neu_col}'", DIM))

    pairs = [{"toxic": r[tox_col], "neutral": r[neu_col]}
             for r in ds
             if r.get(tox_col) and r.get(neu_col)]

    if not pairs:
        print(col("  [WARN] No pairs loaded — check column names above", YELLOW))
        return []

    sample = random.sample(pairs, min(n, len(pairs)))
    print(col(f"  Sampled {len(sample)} pairs", DIM))
    return sample


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Translate with NLLB-200
# ─────────────────────────────────────────────────────────────────────────────

def load_translator(tgt_lang: str):
    from transformers import pipeline
    import torch
    print(col(f"\nLoading NLLB translator ({NLLB_MODEL}) → {tgt_lang} ...", DIM))
    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU" if device == 0 else "CPU"
    print(col(f"  Device: {device_name}", DIM))
    tr = pipeline(
        "translation",
        model=NLLB_MODEL,
        src_lang=NLLB_CODES["en"],
        tgt_lang=NLLB_CODES[tgt_lang],
        device=device,
        max_length=256,
    )
    return tr


def translate_batch(translator, texts, batch_size=8):
    out = []
    for i in range(0, len(texts), batch_size):
        results = translator(texts[i:i+batch_size], batch_size=batch_size)
        out.extend([r["translation_text"] for r in results])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Optional scoring
# ─────────────────────────────────────────────────────────────────────────────

def load_tox_scorer():
    from transformers import pipeline
    import torch
    print(col("Loading toxicity scorer (xlmr-large-toxicity-classifier-v2) ...", DIM))
    device = 0 if torch.cuda.is_available() else -1
    clf = pipeline("text-classification",
                   model="textdetox/xlmr-large-toxicity-classifier-v2",
                   device=device, truncation=True, max_length=512)
    def score(texts):
        results = clf(texts, batch_size=16)
        return [r["score"] if r["label"] == "toxic" else 1 - r["score"]
                for r in results]
    return score


def load_sim_scorer():
    from sentence_transformers import SentenceTransformer, util
    print(col("Loading LaBSE similarity model ...", DIM))
    model = SentenceTransformer("sentence-transformers/LaBSE")
    def score_pairs(texts_a, texts_b):
        ea = model.encode(texts_a, convert_to_tensor=True, batch_size=32, show_progress_bar=False)
        eb = model.encode(texts_b, convert_to_tensor=True, batch_size=32, show_progress_bar=False)
        return util.cos_sim(ea, eb).diagonal().cpu().tolist()
    return score_pairs


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Display + Save
# ─────────────────────────────────────────────────────────────────────────────

def show_pair(i, en_pair, tgt_toxic, tgt_neutral,
              tox_score=None, sim_score=None,
              min_tox=0.5, max_sim=0.92):
    """Print one BT pair with filter verdict."""

    # Determine keep/discard
    tox_ok  = (tox_score is None) or (tox_score >= min_tox)
    sim_ok  = (sim_score is None) or (sim_score <= max_sim)
    keep    = tox_ok and sim_ok

    verdict = col("  KEEP", GREEN + BOLD) if keep else col("  DISCARD", RED + BOLD)
    if not tox_ok:
        verdict += col(f" [tox={tox_score:.3f} < {min_tox} → MT sanitized toxic side]", RED)
    if not sim_ok:
        verdict += col(f" [sim={sim_score:.3f} > {max_sim} → rewrite collapsed]", RED)

    print()
    print(col(f"  ── [{i+1}] ──────────────────────────────────────────────────────", DIM))

    # EN source
    print(col("  EN TOXIC   :", BOLD))
    print(f"    {wrap(en_pair['toxic'])}")
    print(col("  EN NEUTRAL :", BOLD))
    print(f"    {wrap(en_pair['neutral'])}")

    # Translated
    print(col("  → TGT TOXIC   :", CYAN + BOLD))
    print(f"    {wrap(tgt_toxic)}")
    print(col("  → TGT NEUTRAL :", CYAN + BOLD))
    print(f"    {wrap(tgt_neutral)}")

    # Scores
    score_parts = []
    if tox_score is not None:
        tox_col = GREEN if tox_ok else RED
        score_parts.append(col(f"tox_score={tox_score:.3f} ({'≥' if tox_ok else '<'}{min_tox})", tox_col))
    if sim_score is not None:
        sim_col = GREEN if sim_ok else RED
        score_parts.append(col(f"sim(toxic,neutral)={sim_score:.3f} ({'≤' if sim_ok else '>'}{max_sim})", sim_col))
    if score_parts:
        print("  " + "  |  ".join(score_parts))

    print(verdict)


def save_bt_pairs_to_file(
    lang: str,
    en_pairs: list,
    tgt_toxics: list,
    tgt_neutrals: list,
    sim_scores: list,
    max_sim: float,
    out_path: Path,
):
    """
    Write BT pairs to a plain-text file matching the inspect_outputs.py format:

        [1] EN TOXIC    : <text>
            EN NEUTRAL  : <text>
            TGT TOXIC   : <text>
            TGT NEUTRAL : <text>
            sim=X.XXX   KEEP / DISCARD [similarity guard]
    """
    kept   = sum(1 for ss in (sim_scores or []) if ss <= max_sim) if sim_scores else len(en_pairs)
    total  = len(en_pairs)

    lines = []
    lines.append("=" * 92)
    lines.append(f"  BACKTRANSLATION PAIRS  |  Lang: {lang.upper()}  |  Model: {NLLB_MODEL}")
    lines.append(f"  Source: ScaDSAI/ParaDeHate  |  Total pairs: {total}  |  "
                 f"Kept (sim ≤ {max_sim}): {kept}  ({100*kept//total if total else 0}%)")
    lines.append("-" * 92)

    for i, (en_p, tt, tn) in enumerate(zip(en_pairs, tgt_toxics, tgt_neutrals)):
        ss = sim_scores[i] if sim_scores else None
        sim_ok = (ss is None) or (ss <= max_sim)
        verdict = "KEEP" if sim_ok else f"DISCARD [sim={ss:.3f} > {max_sim} → rewrite collapsed]"

        lines.append(f"\n  [{i+1}]")
        lines.append("  EN TOXIC    :")
        for ln in textwrap.wrap(en_p["toxic"], width=80):
            lines.append(f"    {ln}")
        lines.append("  EN NEUTRAL  :")
        for ln in textwrap.wrap(en_p["neutral"], width=80):
            lines.append(f"    {ln}")
        lines.append(f"  → TGT TOXIC ({lang.upper()})   :")
        for ln in textwrap.wrap(tt, width=80):
            lines.append(f"    {ln}")
        lines.append(f"  → TGT NEUTRAL ({lang.upper()}) :")
        for ln in textwrap.wrap(tn, width=80):
            lines.append(f"    {ln}")
        score_str = f"sim(tgt_toxic, tgt_neutral)={ss:.3f}" if ss is not None else "sim=N/A"
        lines.append(f"  {score_str}  →  {verdict}")

    lines.append("\n" + "=" * 92)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(col(f"  Saved {total} BT pairs → {out_path}", GREEN))


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Preview backtranslation output")
    ap.add_argument("--lang", choices=["es", "hi", "both"], default="both",
                    help="Target language (default: both)")
    ap.add_argument("--n", type=int, default=50,
                    help="Number of EN pairs to translate (default: 50)")
    ap.add_argument("--max-sim", type=float, default=0.92,
                    help="Max toxic-neutral similarity for similarity guard (default: 0.92)")
    ap.add_argument("--no-filter-score", action="store_true",
                    help="Skip similarity scoring (just show translations)")
    ap.add_argument("--save-pairs", metavar="DIR", default=None,
                    help="Save BT pairs to DIR/bt_pairs_es.txt and DIR/bt_pairs_hi.txt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    langs = ["es", "hi"] if args.lang == "both" else [args.lang]

    print("\n" + col("=" * 88, BOLD))
    print(col("  PolyDetox — Backtranslation Preview", BOLD + CYAN))
    print(col(f"  Source: ScaDSAI/ParaDeHate  |  Model: {NLLB_MODEL}", DIM))
    print(col(f"  Langs: {langs}  |  n={args.n}  |  max_sim={args.max_sim}", DIM))
    print(col("=" * 88, BOLD))

    # Load EN source pairs once
    en_pairs = load_paradehate(args.n, seed=args.seed)
    if not en_pairs:
        print(col("[ERROR] No pairs loaded from ParaDeHate. Check dataset availability.", RED))
        sys.exit(1)

    # Load similarity scorer (toxicity filter removed — all translated pairs kept)
    sim_scorer = None
    if not args.no_filter_score:
        sim_scorer = load_sim_scorer()

    for lang in langs:
        print("\n" + col(f"\n{'═'*88}", BOLD))
        print(col(f"  TARGET LANGUAGE: {lang.upper()}", BOLD + YELLOW))
        print(col(f"{'═'*88}", BOLD))

        # Translate
        translator = load_translator(lang)
        en_toxics   = [p["toxic"]   for p in en_pairs]
        en_neutrals = [p["neutral"] for p in en_pairs]

        print(col(f"\n  Translating {len(en_toxics)} toxic sentences ...", DIM))
        tgt_toxics = translate_batch(translator, en_toxics)

        print(col(f"  Translating {len(en_neutrals)} neutral sentences ...", DIM))
        tgt_neutrals = translate_batch(translator, en_neutrals)

        # Free translator memory
        del translator
        try:
            import torch, gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        # Score if requested (toxicity filter removed — all translated pairs kept)
        sim_scores = None
        if sim_scorer is not None:
            print(col("  Computing similarity between translated pairs ...", DIM))
            sim_scores = sim_scorer(tgt_toxics, tgt_neutrals)

        # Display
        kept = 0
        for i, (en_p, tt, tn) in enumerate(zip(en_pairs, tgt_toxics, tgt_neutrals)):
            ss = sim_scores[i] if sim_scores else None
            sim_ok = (ss is None) or (ss <= args.max_sim)
            if sim_ok:
                kept += 1
            show_pair(i, en_p, tt, tn, None, ss, None, args.max_sim)

        # Summary
        print()
        print(col(f"{'─'*88}", DIM))
        total = len(en_pairs)
        if sim_scores is not None:
            pct = (100 * kept / total) if total > 0 else 0
            n_sim_fail = sum(1 for ss in sim_scores if ss > args.max_sim)
            print(col(f"  SUMMARY ({lang.upper()}): {kept}/{total} pairs would be KEPT "
                      f"({pct:.0f}% pass rate — similarity guard only)", BOLD))
            print(col(f"  Discarded by similarity guard (rewrite collapsed): {n_sim_fail}",
                      RED if n_sim_fail else DIM))
        else:
            print(col(f"  Showing {total} pairs (scoring skipped — use without --no-filter-score)", DIM))

        # Save to file if requested
        if args.save_pairs:
            out_path = Path(args.save_pairs) / f"bt_pairs_{lang}.txt"
            save_bt_pairs_to_file(
                lang=lang,
                en_pairs=en_pairs,
                tgt_toxics=tgt_toxics,
                tgt_neutrals=tgt_neutrals,
                sim_scores=sim_scores,
                max_sim=args.max_sim,
                out_path=out_path,
            )

    print("\n" + col("  Done.", GREEN + BOLD) + "\n")


if __name__ == "__main__":
    main()
