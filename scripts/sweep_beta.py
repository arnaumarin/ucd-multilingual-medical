"""
Beta sweep for UCD on multilingual medical QA.

Motivation: at beta=1.0 the cumulative energy is dominated by the logit trace
(a sum over all prompt tokens), which is nearly equal for the expert and base
models — so the energy-normalized weights collapse to ~0.5 and UCD degenerates
into static CD. beta discounts older prompt tokens; lowering it should restore
dynamic range to the weights and give UCD's uncertainty signal real influence.

Efficiency: the per-position logits do not depend on beta, so we run each model's
forward pass ONCE per sample, cache the selected-token logit sequence + final
logits, then recompute the trace/energy/weights/prediction for every beta cheaply.

Usage:
    python scripts/sweep_beta.py                       # all 8 langs, default betas
    python scripts/sweep_beta.py --n 120 --betas 0 0.9 1.0 --languages en ar
"""

import argparse, json, math, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).parent))
from ucd_engine import (
    format_mc_prompt, get_answer_token_ids, decide_answer_form, align_vocab, ANSWER_CHOICES,
)
from run_fast import LANGUAGES, MEDICAL_SUBJECTS, load_samples, MODEL_EXP, MODEL_BASE, pick_device

OUT_DIR = Path(__file__).parents[1] / "results"


def cache_sample(model, tokenizer, prompt):
    """One forward pass. Return (final_logits[V], selected_logit_sequence[L], lse)."""
    ids = tokenizer.encode(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        all_logits = model(ids).logits[0].float()      # [seq, V]
    final = all_logits[-1]                              # [V]
    # selected logit at each transition: logits[k, ids[k+1]]
    nxt = ids[0, 1:]                                    # [seq-1]
    sel = all_logits[:-1].gather(1, nxt.unsqueeze(1)).squeeze(1)  # [seq-1]
    lse = torch.logsumexp(final, dim=0).item()
    return final, sel, lse


def trace_for_beta(sel: torch.Tensor, beta: float) -> float:
    """Discounted sum with the most-recent token weighted beta^0 = 1 (eq. 2)."""
    if beta == 1.0:
        return float(sel.sum())
    # vectorized: weights beta^(L-1-k) for k=0..L-1
    L = sel.shape[0]
    powers = beta ** torch.arange(L - 1, -1, -1, dtype=torch.float32, device=sel.device)
    return float((sel * powers).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--betas", type=float, nargs="+",
                    default=[0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0])
    ap.add_argument("--languages", nargs="+", default=list(LANGUAGES.keys()))
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    dev = pick_device()
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    print(f"Device: {dev} | dtype: {dtype} | betas: {args.betas}")

    tok = AutoTokenizer.from_pretrained(MODEL_EXP)
    expert = AutoModelForCausalLM.from_pretrained(MODEL_EXP, dtype=dtype, device_map=dev).eval()
    base = AutoModelForCausalLM.from_pretrained(MODEL_BASE, dtype=dtype, device_map=dev).eval()
    answer_ids = get_answer_token_ids(tok)

    methods = ["greedy", "cd", "ucd"]
    # acc[beta][lang][method] -> [correct, total];  wstats[beta] -> list of w_exp
    acc = {b: {l: {m: [0, 0] for m in methods} for l in args.languages} for b in args.betas}
    wstats = {b: [] for b in args.betas}

    for lang in args.languages:
        samples = load_samples(lang, LANGUAGES[lang], args.n)
        t0 = time.time()
        for item in samples:
            prompt = format_mc_prompt(item["question"], item["choices"], lang)
            ef, es, e_lse = cache_sample(expert, tok, prompt)
            bf, bs, b_lse = cache_sample(base, tok, prompt)
            ef, bf = align_vocab(ef, bf)  # different lm_head widths across sizes
            gold = item["answer"]

            chosen, _ = decide_answer_form(ef, answer_ids)
            cidx = {L: chosen[L] for L in ANSWER_CHOICES}

            def pick(vec):
                return max(cidx, key=lambda L: vec[cidx[L]].item())

            greedy_pred = pick(ef)
            cd_pred = pick(2 * ef - bf)

            for b in args.betas:
                Ee = trace_for_beta(es, b) + e_lse
                Eb = trace_for_beta(bs, b) + b_lse
                if Ee > 0 and Eb > 0:
                    we = Ee / (Ee + Eb)
                    wb = Eb / (Ee + Eb)
                    ucd_logits = (1 + args.alpha) * we * ef - wb * bf
                    wstats[b].append(we)
                else:
                    ucd_logits = ef
                ucd_pred = pick(ucd_logits)

                for m, pred in [("greedy", greedy_pred), ("cd", cd_pred), ("ucd", ucd_pred)]:
                    acc[b][lang][m][0] += int(pred == gold)
                    acc[b][lang][m][1] += 1
        print(f"[{lang}] {len(samples)} samples  ({time.time()-t0:.1f}s)")

    # ---- summarize ----
    summary = {}
    for b in args.betas:
        summary[str(b)] = {}
        for lang in args.languages:
            summary[str(b)][lang] = {
                m: acc[b][lang][m][0] / acc[b][lang][m][1] for m in methods
            }

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"beta_sweep_0.5B_n{args.n}.json"
    with open(out, "w") as f:
        json.dump({"summary": summary,
                   "w_exp_stats": {str(b): {"mean": float(np.mean(w)),
                                            "std": float(np.std(w)),
                                            "min": float(np.min(w)),
                                            "max": float(np.max(w))}
                                   for b, w in wstats.items() if w}}, f, indent=2)

    # ---- print table: avg-over-languages accuracy + weight spread per beta ----
    print("\n" + "=" * 72)
    print(f"{'beta':>6} {'greedy':>8} {'cd':>8} {'ucd':>8} {'UCD-gr':>8}   "
          f"{'w_exp mean':>10} {'w_exp std':>9}")
    print("-" * 72)
    for b in args.betas:
        avg = {m: float(np.mean([summary[str(b)][l][m] for l in args.languages])) for m in methods}
        w = wstats[b]
        wm, ws = (np.mean(w), np.std(w)) if w else (float("nan"), float("nan"))
        print(f"{b:>6} {avg['greedy']:>8.3f} {avg['cd']:>8.3f} {avg['ucd']:>8.3f} "
              f"{avg['ucd']-avg['greedy']:>+8.3f}   {wm:>10.4f} {ws:>9.4f}")
    print("=" * 72)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
