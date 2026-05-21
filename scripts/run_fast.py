"""
Efficient UCD experiment runner.
Computes expert + base logits once per sample, applies greedy / CD / UCD in one pass.
Saves per-item records (with subject, language, energies) for all downstream plots.
"""

import json, sys, time
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from ucd_engine import (
    UCDConfig, format_mc_prompt, evaluate_mc_sample, get_answer_token_ids,
    ANSWER_CHOICES,
)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# ── Config ──────────────────────────────────────────────────────────────────
MEDICAL_SUBJECTS = [
    "clinical_knowledge", "medical_genetics", "anatomy",
    "professional_medicine", "college_medicine", "college_biology",
    "high_school_biology", "nutrition", "virology",
]

LANGUAGES = {
    "en": "default",
    "zh": "ZH_CN",
    "es": "ES_LA",
    "fr": "FR_FR",
    "de": "DE_DE",
    "ar": "AR_XY",
    "ko": "KO_KR",
    "ja": "JA_JP",
}

N_SAMPLES  = 150      # per language (stratified across subjects)
MODEL_EXP  = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_BASE = "Qwen/Qwen2.5-0.5B"
OUT_DIR    = Path(__file__).parents[1] / "results"
DEVICE     = pick_device()
# fp32 on CPU/MPS for stability; bf16 on CUDA for speed (logit math is re-cast
# to fp32 inside the engine regardless).
DTYPE      = torch.bfloat16 if DEVICE == "cuda" else torch.float32


# ── Dataset helpers ──────────────────────────────────────────────────────────
def load_samples(lang_code, mmmlu_config, n):
    if mmmlu_config == "default":
        rows = []
        for subj in MEDICAL_SUBJECTS:
            ds = load_dataset("cais/mmlu", subj, split="test")
            for item in ds:
                rows.append({
                    "question": item["question"],
                    "choices":  item["choices"],
                    "answer":   ["A","B","C","D"][item["answer"]],
                    "subject":  subj,
                    "lang":     lang_code,
                })
    else:
        ds = load_dataset("openai/MMMLU", mmmlu_config, split="test")
        rows = [
            {"question": r["Question"],
             "choices":  [r["A"], r["B"], r["C"], r["D"]],
             "answer":   r["Answer"],
             "subject":  r["Subject"],
             "lang":     lang_code}
            for r in ds if r["Subject"] in MEDICAL_SUBJECTS
        ]

    # Stratified sample: equal per subject
    import random, math
    random.seed(42)
    per_subj = math.ceil(n / len(MEDICAL_SUBJECTS))
    by_subj = {}
    for r in rows:
        by_subj.setdefault(r["subject"], []).append(r)
    sampled = []
    for subj_rows in by_subj.values():
        random.shuffle(subj_rows)
        sampled.extend(subj_rows[:per_subj])
    random.shuffle(sampled)
    return sampled[:n]


# ── Core per-sample evaluation ───────────────────────────────────────────────
def evaluate_sample(expert, base, tokenizer, item, config, answer_ids):
    """Thin wrapper over the shared engine core; adds lang/subject + rounds."""
    prompt = format_mc_prompt(item["question"], item["choices"], item["lang"])
    r = evaluate_mc_sample(expert, base, tokenizer, prompt,
                           item["answer"], config, answer_ids)
    return {
        "lang":         item["lang"],
        "subject":      item["subject"],
        "gold":         r["gold"],
        "greedy_pred":  r["greedy_pred"],
        "cd_pred":      r["cd_pred"],
        "ucd_pred":     r["ucd_pred"],
        "greedy_ok":    r["greedy_ok"],
        "cd_ok":        r["cd_ok"],
        "ucd_ok":       r["ucd_ok"],
        "exp_energy":   round(r["exp_energy"],  4),
        "base_energy":  round(r["base_energy"], 4),
        "exp_trace":    round(r["exp_trace"],   4),
        "base_trace":   round(r["base_trace"],  4),
        "answer_form":  r["answer_form"],
        "greedy_scores": {k: round(v, 4) for k, v in r["greedy_scores"].items()},
        "ucd_scores":    {k: round(v, 4) for k, v in r["ucd_scores"].items()},
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Fast UCD multilingual-medical runner.")
    p.add_argument("--expert", default=MODEL_EXP, help="Expert (instruct) model id.")
    p.add_argument("--base", default=MODEL_BASE, help="Base (amateur) model id.")
    p.add_argument("--tag", default="0.5B",
                   help="Output tag → records_<tag>_n<N>.json / summary_<tag>_n<N>.json")
    p.add_argument("--n", type=int, default=N_SAMPLES, help="Samples per language.")
    p.add_argument("--languages", nargs="+", default=list(LANGUAGES.keys()))
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = UCDConfig(beta=args.beta, temperature=args.temperature)

    print(f"Device: {DEVICE} | dtype: {DTYPE}")
    print(f"Loading expert  : {args.expert}")
    tokenizer = AutoTokenizer.from_pretrained(args.expert)
    expert    = AutoModelForCausalLM.from_pretrained(
        args.expert, dtype=DTYPE, device_map=DEVICE).eval()

    print(f"Loading base    : {args.base}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=DTYPE, device_map=DEVICE).eval()

    answer_ids = get_answer_token_ids(tokenizer)
    print(f"Answer token IDs: {answer_ids}\n")

    all_records = []
    summary = {}

    for lang_code in args.languages:
        mmmlu_cfg = LANGUAGES[lang_code]
        print(f"[{lang_code.upper()}] Loading {mmmlu_cfg}...")
        samples = load_samples(lang_code, mmmlu_cfg, args.n)
        print(f"  {len(samples)} samples | running inference...")

        lang_records = []
        t0 = time.time()
        for item in tqdm(samples, desc=f"  {lang_code}", leave=False):
            rec = evaluate_sample(expert, base, tokenizer, item, config, answer_ids)
            lang_records.append(rec)
        elapsed = time.time() - t0

        # Compute accuracy per method
        n = len(lang_records)
        accs = {
            m: sum(r[f"{m}_ok"] for r in lang_records) / n
            for m in ["greedy", "cd", "ucd"]
        }
        summary[lang_code] = accs
        all_records.extend(lang_records)

        acc_str = "  ".join(f"{m}: {v:.3f}" for m, v in accs.items())
        print(f"  [{lang_code}] {acc_str}  ({elapsed:.1f}s)")

    # Save
    records_path = OUT_DIR / f"records_{args.tag}_n{args.n}.json"
    with open(records_path, "w") as f:
        json.dump(all_records, f)
    print(f"\nRecords saved → {records_path}")

    summary_path = OUT_DIR / f"summary_{args.tag}_n{args.n}.json"
    with open(summary_path, "w") as f:
        json.dump({"expert": args.expert, "base": args.base, "summary": summary}, f, indent=2)
    print(f"Summary saved  → {summary_path}")

    # Print table
    print("\n" + "="*55)
    print(f"{'Lang':8} {'Greedy':>8} {'CD':>8} {'UCD':>8}")
    print("-"*55)
    for lang, accs in summary.items():
        print(f"{lang:8} {accs['greedy']:>8.3f} {accs['cd']:>8.3f} {accs['ucd']:>8.3f}")
    import numpy as np
    avgs = {m: float(np.mean([summary[l][m] for l in summary])) for m in ["greedy","cd","ucd"]}
    print("-"*55)
    print(f"{'AVG':8} {avgs['greedy']:>8.3f} {avgs['cd']:>8.3f} {avgs['ucd']:>8.3f}")
    print("="*55)


if __name__ == "__main__":
    main()
