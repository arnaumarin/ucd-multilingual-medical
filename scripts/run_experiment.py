"""
Main experiment runner: UCD multilingual medical QA evaluation.
Runs greedy / CD / UCD across 8 languages on MMMLU medical subjects.

Usage:
    python run_experiment.py --model_size 0.5B --n_samples 200 --output_dir ../results
    python run_experiment.py --model_size 1.5B --n_samples 500 --output_dir ../results
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ucd_engine import (
    UCDConfig, format_mc_prompt, evaluate_mc_sample, get_answer_token_ids,
)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MEDICAL_SUBJECTS = [
    "clinical_knowledge",
    "medical_genetics",
    "anatomy",
    "professional_medicine",
    "college_medicine",
    "college_biology",
    "high_school_biology",
    "nutrition",
    "virology",
]

LANGUAGES = {
    "en": "default",      # original English MMLU
    "zh": "ZH_CN",
    "es": "ES_LA",
    "fr": "FR_FR",
    "de": "DE_DE",
    "ar": "AR_XY",
    "ko": "KO_KR",
    "ja": "JA_JP",
}

# Qwen2.5 is chosen for strong multilingual coverage across all 8 target languages
MODEL_CONFIGS = {
    "0.5B": {
        "expert": "Qwen/Qwen2.5-0.5B-Instruct",
        "base":   "Qwen/Qwen2.5-0.5B",
    },
    "1.5B": {
        "expert": "Qwen/Qwen2.5-1.5B-Instruct",
        "base":   "Qwen/Qwen2.5-1.5B",
    },
    "7B": {
        "expert": "Qwen/Qwen2.5-7B-Instruct",
        "base":   "Qwen/Qwen2.5-7B",
    },
}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_mmmlu_medical(lang_code: str, mmmlu_config: str, n_samples: int) -> list[dict]:
    """Load and filter MMMLU to medical subjects for a given language."""
    if mmmlu_config == "default":
        # English: load from cais/mmlu per subject (original, non-translated)
        rows = []
        for subj in MEDICAL_SUBJECTS:
            ds = load_dataset("cais/mmlu", subj, split="test")
            for item in ds:
                rows.append({
                    "question": item["question"],
                    "choices":  item["choices"],
                    "answer":   ["A", "B", "C", "D"][item["answer"]],
                    "subject":  subj,
                    "lang":     lang_code,
                })
    else:
        ds = load_dataset("openai/MMMLU", mmmlu_config, split="test")
        rows = []
        for item in ds:
            if item["Subject"] in MEDICAL_SUBJECTS:
                rows.append({
                    "question": item["Question"],
                    "choices":  [item["A"], item["B"], item["C"], item["D"]],
                    "answer":   item["Answer"],
                    "subject":  item["Subject"],
                    "lang":     lang_code,
                })

    if n_samples and n_samples < len(rows):
        # Stratified sample: take an equal number per subject so every medical
        # subject is represented (not just a random slice of the pooled rows).
        import random, math
        random.seed(42)
        per_subj = math.ceil(n_samples / len(MEDICAL_SUBJECTS))
        by_subj: dict[str, list] = {}
        for r in rows:
            by_subj.setdefault(r["subject"], []).append(r)
        sampled = []
        for subj_rows in by_subj.values():
            random.shuffle(subj_rows)
            sampled.extend(subj_rows[:per_subj])
        random.shuffle(sampled)
        rows = sampled[:n_samples]

    return rows


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_model(model_name: str, device: str = "cpu", dtype=torch.float32):
    """Load model and tokenizer. bf16 on CUDA, fp32 on CPU/MPS."""
    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_language(
    lang_code: str,
    samples: list[dict],
    expert_model,
    base_model,
    tokenizer,
    config: UCDConfig,
    methods: list[str],
    answer_ids: dict,
) -> dict:
    """Run all methods on a language's samples in a single pair of forward passes
    per sample. Returns per-method accuracy + per-sample energy records."""
    results = {m: {"correct": 0, "total": 0, "energies": []} for m in methods}

    for item in tqdm(samples, desc=f"  {lang_code}", leave=False):
        prompt = format_mc_prompt(item["question"], item["choices"], lang_code)
        r = evaluate_mc_sample(
            expert_model, base_model, tokenizer, prompt,
            item["answer"], config, answer_ids,
        )

        for method in methods:
            is_correct = r[f"{method}_ok"]
            results[method]["correct"] += int(is_correct)
            results[method]["total"] += 1
            results[method]["energies"].append({
                "exp":     r["exp_energy"],
                "base":    r["base_energy"],
                "lang":    lang_code,
                "subject": item["subject"],
                "correct": is_correct,
            })

    for method in methods:
        n = results[method]["total"]
        results[method]["accuracy"] = results[method]["correct"] / n if n > 0 else 0.0

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", default="0.5B", choices=["0.5B", "1.5B", "7B"])
    parser.add_argument("--n_samples", type=int, default=200,
                        help="Samples per language (0 = all)")
    parser.add_argument("--output_dir", default="../results")
    parser.add_argument("--languages", nargs="+", default=list(LANGUAGES.keys()),
                        help="Language codes to run (default: all)")
    parser.add_argument("--methods", nargs="+", default=["greedy", "cd", "ucd"])
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = UCDConfig(beta=args.beta, temperature=args.temperature)
    model_cfg = MODEL_CONFIGS[args.model_size]
    device = pick_device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Device: {device} | dtype: {dtype}")

    print(f"\nLoading expert model: {model_cfg['expert']}")
    expert_model, tokenizer = load_model(model_cfg["expert"], device, dtype)

    print(f"Loading base model: {model_cfg['base']}")
    base_model, _ = load_model(model_cfg["base"], device, dtype)

    answer_ids = get_answer_token_ids(tokenizer)

    all_results = {}
    all_energy_records = []

    for lang_code in args.languages:
        mmmlu_config = LANGUAGES[lang_code]
        print(f"\n[{lang_code.upper()}] Loading dataset ({mmmlu_config})...")
        samples = load_mmmlu_medical(lang_code, mmmlu_config, args.n_samples)
        print(f"  {len(samples)} samples loaded")

        start = time.time()
        lang_results = evaluate_language(
            lang_code, samples, expert_model, base_model,
            tokenizer, config, args.methods, answer_ids,
        )
        elapsed = time.time() - start

        all_results[lang_code] = lang_results
        for method in args.methods:
            for e in lang_results[method]["energies"]:
                e["method"] = method
                all_energy_records.append(e)

        # Print live results
        acc_str = " | ".join(
            f"{m}: {lang_results[m]['accuracy']:.3f}" for m in args.methods
        )
        print(f"  {lang_code}: {acc_str}  ({elapsed:.1f}s)")

    # Save outputs
    out_file = out_dir / f"results_{args.model_size}_n{args.n_samples}.json"
    with open(out_file, "w") as f:
        # Remove non-serializable energy lists from main results
        clean = {}
        for lang, lang_res in all_results.items():
            clean[lang] = {}
            for method, stats in lang_res.items():
                clean[lang][method] = {
                    "accuracy": stats["accuracy"],
                    "correct":  stats["correct"],
                    "total":    stats["total"],
                }
        json.dump({
            "config": vars(args),
            "results": clean,
        }, f, indent=2)
    print(f"\nResults saved: {out_file}")

    energy_file = out_dir / f"energies_{args.model_size}_n{args.n_samples}.json"
    with open(energy_file, "w") as f:
        json.dump(all_energy_records, f)
    print(f"Energy records saved: {energy_file}")

    # Print summary table
    print("\n" + "="*70)
    print(f"SUMMARY — Model: Qwen2.5-{args.model_size}")
    print("="*70)
    header = f"{'Lang':6}" + "".join(f"  {m:8}" for m in args.methods)
    print(header)
    print("-"*70)
    for lang in args.languages:
        row = f"{lang:6}"
        for method in args.methods:
            acc = all_results[lang][method]["accuracy"]
            row += f"  {acc:.3f}   "
        print(row)
    print("="*70)


if __name__ == "__main__":
    main()
