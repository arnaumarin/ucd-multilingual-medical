"""
Compare UCD/CD/greedy across model pairings (different expert/base capability gaps).

Reads the per-item records_<tag>_n<N>.json files written by run_fast.py and reports,
per pairing: avg accuracy by method, UCD/CD deltas over greedy, the expert/base
energy correlation (how distinguishable the two models are — the precondition for
contrastive decoding to help), the UCD weight spread, and whether UCD's gain
correlates with language resource level (the project's multilingual hypothesis).

Usage:
    python scripts/compare_pairings.py 0.5B 7Bexp_0.5Bbase 7Bexp_7Bbase --n 150
"""

import argparse, json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).parents[1] / "results"
LANGS = ["en", "zh", "es", "fr", "de", "ar", "ko", "ja"]
RESOURCE = {"en": 5, "fr": 4, "de": 4, "es": 4, "zh": 3, "ja": 3, "ko": 3, "ar": 2}
METHODS = ["greedy", "cd", "ucd"]


def load(tag, n):
    f = RESULTS / f"records_{tag}_n{n}.json"
    if not f.exists():
        return None
    return json.load(open(f))


def acc_by_lang(recs):
    out = {}
    for lang in sorted({r["lang"] for r in recs}):
        rows = [r for r in recs if r["lang"] == lang]
        out[lang] = {m: np.mean([r[f"{m}_ok"] for r in rows]) for m in METHODS}
    return out


def report(tag, recs):
    print("\n" + "=" * 70)
    print(f"PAIRING: {tag}   ({len(recs)} items)")
    print("=" * 70)
    by_lang = acc_by_lang(recs)
    langs = [l for l in LANGS if l in by_lang]

    print(f"{'lang':5} {'greedy':>8} {'cd':>8} {'ucd':>8} {'cd-gr':>7} {'ucd-gr':>7}")
    for lang in langs:
        a = by_lang[lang]
        print(f"{lang:5} {a['greedy']:>8.3f} {a['cd']:>8.3f} {a['ucd']:>8.3f} "
              f"{a['cd']-a['greedy']:>+7.3f} {a['ucd']-a['greedy']:>+7.3f}")
    avg = {m: np.mean([by_lang[l][m] for l in langs]) for m in METHODS}
    print("-" * 56)
    print(f"{'AVG':5} {avg['greedy']:>8.3f} {avg['cd']:>8.3f} {avg['ucd']:>8.3f} "
          f"{avg['cd']-avg['greedy']:>+7.3f} {avg['ucd']-avg['greedy']:>+7.3f}")

    # contrast precondition: how distinguishable are expert and base?
    ee = np.array([r["exp_energy"] for r in recs])
    be = np.array([r["base_energy"] for r in recs])
    valid = (ee + be) != 0
    w = ee[valid] / (ee + be)[valid]
    corr = np.corrcoef(ee, be)[0, 1]
    print(f"\ncorr(exp_E, base_E) = {corr:.4f}   "
          f"w_exp = {w.mean():.3f} ± {w.std():.3f}   "
          f"(corr→1 means models too similar for contrast to help)")

    # multilingual hypothesis: does UCD gain grow as resource level drops?
    gains = [by_lang[l]["ucd"] - by_lang[l]["greedy"] for l in langs]
    res = [RESOURCE[l] for l in langs]
    if len(set(res)) > 1:
        r = np.corrcoef(res, gains)[0, 1]
        print(f"corr(resource_level, UCD_gain) = {r:+.3f}   "
              f"(hypothesis wants NEGATIVE: bigger gain for lower-resource langs)")
    return {"tag": tag, "avg": avg, "corr_energy": corr,
            "w_mean": float(w.mean()), "w_std": float(w.std())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()

    rows = []
    for tag in args.tags:
        recs = load(tag, args.n)
        if recs is None:
            print(f"[skip] no records for tag '{tag}' (n={args.n})")
            continue
        rows.append(report(tag, recs))

    if len(rows) > 1:
        print("\n" + "#" * 70)
        print("SIDE-BY-SIDE")
        print("#" * 70)
        print(f"{'pairing':22} {'greedy':>7} {'cd':>7} {'ucd':>7} "
              f"{'ucd-gr':>7} {'corrE':>7} {'w_std':>7}")
        for r in rows:
            a = r["avg"]
            print(f"{r['tag']:22} {a['greedy']:>7.3f} {a['cd']:>7.3f} {a['ucd']:>7.3f} "
                  f"{a['ucd']-a['greedy']:>+7.3f} {r['corr_energy']:>7.3f} {r['w_std']:>7.3f}")


if __name__ == "__main__":
    main()
