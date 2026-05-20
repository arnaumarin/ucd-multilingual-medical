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
    UCDConfig, format_mc_prompt,
    compute_logit_trace, compute_energy, ucd_score, get_answer_token_ids,
    ANSWER_CHOICES,
)

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
DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"


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
    prompt    = format_mc_prompt(item["question"], item["choices"], item["lang"])
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Single forward pass each
    exp_logits, exp_trace = compute_logit_trace(expert, tokenizer, input_ids,
                                                config.beta, config.temperature)
    base_logits, base_trace = compute_logit_trace(base, tokenizer, input_ids,
                                                  config.beta, config.temperature)

    exp_energy  = compute_energy(exp_logits,  exp_trace,  config.temperature)
    base_energy = compute_energy(base_logits, base_trace, config.temperature)

    # UCD logits
    ucd_logits = ucd_score(exp_logits, base_logits, exp_energy, base_energy, alpha=1.0)

    # CD logits (static equal weighting)
    cd_logits = 2 * exp_logits - base_logits

    gold = item["answer"]

    def pick(logit_vec):
        scores = {ch: logit_vec[tid].item() for ch, tid in answer_ids.items()}
        return max(scores, key=scores.get), scores

    greedy_pred, greedy_scores = pick(exp_logits)
    cd_pred,     cd_scores     = pick(cd_logits)
    ucd_pred,    ucd_scores    = pick(ucd_logits)

    return {
        "lang":         item["lang"],
        "subject":      item["subject"],
        "gold":         gold,
        "greedy_pred":  greedy_pred,
        "cd_pred":      cd_pred,
        "ucd_pred":     ucd_pred,
        "greedy_ok":    greedy_pred == gold,
        "cd_ok":        cd_pred     == gold,
        "ucd_ok":       ucd_pred    == gold,
        "exp_energy":   round(exp_energy,  4),
        "base_energy":  round(base_energy, 4),
        "exp_trace":    round(float(exp_trace),  4),
        "base_trace":   round(float(base_trace), 4),
        "greedy_scores": {k: round(v,4) for k,v in greedy_scores.items()},
        "ucd_scores":    {k: round(v,4) for k,v in ucd_scores.items()},
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = UCDConfig()

    print(f"Device: {DEVICE}")
    print(f"Loading expert  : {MODEL_EXP}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_EXP)
    expert    = AutoModelForCausalLM.from_pretrained(
        MODEL_EXP, dtype=torch.float32, device_map=DEVICE).eval()

    print(f"Loading base    : {MODEL_BASE}")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_BASE, dtype=torch.float32, device_map=DEVICE).eval()

    answer_ids = get_answer_token_ids(tokenizer)
    print(f"Answer token IDs: {answer_ids}\n")

    all_records = []
    summary = {}

    for lang_code, mmmlu_cfg in LANGUAGES.items():
        print(f"[{lang_code.upper()}] Loading {mmmlu_cfg}...")
        samples = load_samples(lang_code, mmmlu_cfg, N_SAMPLES)
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
    records_path = OUT_DIR / "records_0.5B_n150.json"
    with open(records_path, "w") as f:
        json.dump(all_records, f)
    print(f"\nRecords saved → {records_path}")

    summary_path = OUT_DIR / "summary_0.5B_n150.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
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
