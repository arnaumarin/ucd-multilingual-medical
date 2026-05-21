"""
Generation-mode (multi-token / cloze) UCD experiment runner.

Instead of scoring a single answer-letter token, this scores each answer CHOICE's
full TEXT as a continuation of the question (MC2/MC3-style), token-by-token, under
greedy / CD / UCD. This is the regime contrastive decoding was designed for: the
logit trace accumulates over the generated answer and the energy weight varies per
token. Tests whether CD/UCD help when scoring is multi-token, where single-letter
MC showed greedy winning everywhere.

Usage (parallels run_fast.py):
    python scripts/run_generation.py --expert Qwen/Qwen2.5-7B-Instruct --base Qwen/Qwen2.5-7B --tag 7Bexp_7Bbase
"""

import json, sys, time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from ucd_engine import UCDConfig, format_cloze_context, score_answer_text_generation
from run_fast import LANGUAGES, load_samples, MODEL_EXP, MODEL_BASE, pick_device

OUT_DIR = Path(__file__).parents[1] / "results"
DEVICE = pick_device()
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
METHODS = ["greedy", "cd", "ucd"]


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Generation-mode UCD runner.")
    p.add_argument("--expert", default=MODEL_EXP)
    p.add_argument("--base", default=MODEL_BASE)
    p.add_argument("--tag", default="0.5B")
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--languages", nargs="+", default=list(LANGUAGES.keys()))
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = UCDConfig(beta=args.beta, temperature=args.temperature)

    print(f"Device: {DEVICE} | dtype: {DTYPE}")
    print(f"Loading expert : {args.expert}")
    tok = AutoTokenizer.from_pretrained(args.expert)
    expert = AutoModelForCausalLM.from_pretrained(args.expert, dtype=DTYPE, device_map=DEVICE).eval()
    print(f"Loading base   : {args.base}")
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=DTYPE, device_map=DEVICE).eval()

    all_records, summary = [], {}
    for lang in args.languages:
        samples = load_samples(lang, LANGUAGES[lang], args.n)
        print(f"[{lang.upper()}] {len(samples)} samples | scoring...")
        recs = []
        t0 = time.time()
        for item in tqdm(samples, desc=f"  {lang}", leave=False):
            ctx = format_cloze_context(item["question"], lang)
            preds, _ = score_answer_text_generation(expert, base, tok, ctx, item["choices"], cfg)
            gold_idx = "ABCD".index(item["answer"])
            recs.append({
                "lang": lang, "subject": item["subject"], "gold": item["answer"],
                **{f"{m}_ok": (preds[m] == gold_idx) for m in METHODS},
                **{f"{m}_pred": "ABCD"[preds[m]] for m in METHODS},
            })
        accs = {m: sum(r[f"{m}_ok"] for r in recs) / len(recs) for m in METHODS}
        summary[lang] = accs
        all_records.extend(recs)
        print(f"  [{lang}] " + "  ".join(f"{m}: {accs[m]:.3f}" for m in METHODS)
              + f"  ({time.time()-t0:.1f}s)")

    rec_path = OUT_DIR / f"records_gen_{args.tag}_n{args.n}.json"
    json.dump(all_records, open(rec_path, "w"))
    sum_path = OUT_DIR / f"summary_gen_{args.tag}_n{args.n}.json"
    json.dump({"expert": args.expert, "base": args.base, "mode": "generation", "summary": summary},
              open(sum_path, "w"), indent=2)
    print(f"\nRecords → {rec_path}\nSummary → {sum_path}")

    import numpy as np
    print("\n" + "=" * 55)
    print(f"{'Lang':8} {'Greedy':>8} {'CD':>8} {'UCD':>8}")
    print("-" * 55)
    for lang, a in summary.items():
        print(f"{lang:8} {a['greedy']:>8.3f} {a['cd']:>8.3f} {a['ucd']:>8.3f}")
    avg = {m: float(np.mean([summary[l][m] for l in summary])) for m in METHODS}
    print("-" * 55)
    print(f"{'AVG':8} {avg['greedy']:>8.3f} {avg['cd']:>8.3f} {avg['ucd']:>8.3f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
