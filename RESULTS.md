# Results — Does UCD Help Multilingual Medical Multiple-Choice QA?

**TL;DR.** Across four expert/base pairings spanning model **scale**, **capability gap**, and **domain gap**, plain **greedy (expert-only) decoding beats both Contrastive Decoding (CD) and Uncertainty-Aware Contrastive Decoding (UCD)** on MMMLU-Medical single-step multiple-choice across 8 languages. UCD ≈ CD everywhere — the energy-based dynamic weighting adds nothing measurable. Critically, when the contrast is made genuinely "live" (expert/base energies de-correlate), it still doesn't help, and a **true medical domain gap makes accuracy ~6 points *worse***. We attribute this to contrastive decoding being a **generation-time** mechanism that does not transfer to single-token MC answer selection.

A follow-up **generation-mode** experiment (§6, multi-token cloze scoring) supports that attribution: CD's effect on accuracy *flips from negative to positive* when scoring becomes multi-token — but **UCD's dynamic weighting still adds no reliable benefit over static CD**, and where it does help it helps *high*-resource languages, opposite the hypothesis.

This is a negative result, reported in full.

---

## 1. Setup

| | |
|---|---|
| **Data** | MMMLU-Medical — `openai/MMMLU` for non-English, `cais/mmlu` for English. 9 medical subjects (clinical knowledge, medical genetics, anatomy, professional/college medicine, college/HS biology, nutrition, virology). **150 stratified samples/language × 8 languages = 1,200 items per pairing.** |
| **Languages** | en, zh, es, fr, de, ar, ko, ja (resource level 5→2) |
| **Methods** | **Greedy** (expert argmax) · **CD** (`2·z_exp − z_base`) · **UCD** (energy-weighted, eq. 4 of Lee et al. 2025) |
| **Scoring** | Single-step: score the answer-letter token at the `Answer:` position. Surface form (bare `A` vs space-prefixed `␣A`) chosen per-prompt from the expert distribution; Latin A/B/C/D markers in all languages (matches MMMLU gold). |
| **Hardware** | Single RTX 5090, bf16 weights, energy/trace math in fp32. β = T = 1.0. |

**Pairings (expert + base):**

| Tag | Expert | Base | Probes |
|---|---|---|---|
| `0.5B` | Qwen2.5-0.5B-Instruct | Qwen2.5-0.5B | small scale, instruct-vs-base |
| `7Bexp_0.5Bbase` | Qwen2.5-7B-Instruct | Qwen2.5-0.5B | large **capability gap** |
| `7Bexp_7Bbase` | Qwen2.5-7B-Instruct | Qwen2.5-7B | same-size instruct-vs-base (paper-style) |
| `apollo2_qwen7b` | **Apollo2-7B** (multilingual-medical) | Qwen2.5-7B | true **domain gap** |

Apollo2-7B is a Qwen2-based multilingual-medical model; its tokenizer is **identical** to Qwen2.5-7B's (vocab 151643, verified token-for-token), so the contrast operates over the same token ids. (Meditron-7B and Meta LLaMA-2 were gated; BioMistral shipped only as a stalled 28GB fp32 blob — Apollo2 is the ungated, multilingual, tokenizer-compatible equivalent.)

---

## 2. Main result

Accuracy averaged over the 8 languages (n = 1,200/pairing):

| Pairing | Greedy | CD | UCD | UCD−Greedy | corr(E_exp,E_base) | w_exp std | Contrast "live"? |
|---|---|---|---|---|---|---|---|
| `0.5B` | **0.316** | 0.302 | 0.303 | −0.013 | 0.999 | 0.006 | no (weights collapsed) |
| `7Bexp_0.5Bbase` | **0.671** | 0.670 | 0.669 | −0.002 | 0.988 | 0.022 | **yes** |
| `7Bexp_7Bbase` | **0.671** | 0.665 | 0.665 | −0.006 | 0.999 | 0.006 | no |
| `apollo2_qwen7b` | **0.643** | 0.583 | 0.584 | **−0.058** | 0.989 | 0.022 | **yes (domain)** |

**Greedy wins in every regime.** UCD never beats greedy, and **UCD ≡ CD** to within ≤0.001–0.006 in all four pairings — the dynamic weighting is inert for this task.

---

## 3. Mechanism — why the contrast doesn't help (and the experiment that proves it)

**(a) The dynamic weight collapses when the two models are similar.** UCD weights the expert by `w_exp = E_exp / (E_exp + E_base)`. With β = 1, the cumulative energy is **99% the logit trace** (a sum over all prompt tokens). For similar models the expert/base energies are nearly identical (`corr = 0.999`), so `w_exp = 0.4965 ± 0.006` — a constant ≈ 0.5. UCD degenerates into static CD.

**(b) β doesn't rescue it.** A β-sweep (0.0 → 1.0) on the 0.5B pairing restores weight variance (std 0.006 → 0.034 as β → 0) but **accuracy is flat** (UCD stays −0.013 vs greedy at every β). The weighting mechanism was never the bottleneck.

**(c) A "live" contrast still doesn't help — and a domain gap hurts most.** This is the decisive test. In the **capability-gap** (`7Bexp_0.5Bbase`) and **domain-gap** (`apollo2_qwen7b`) pairings the energies genuinely de-correlate (`corr ≈ 0.989`) and the weight variance more than triples (`w_std ≈ 0.022`) — the contrast is fully engaged. Yet:
- the capability gap still doesn't beat greedy (−0.002), and
- the **domain gap hurts the most of any pairing (−0.058)**.

The more the expert and base genuinely differ in *what they know*, the more the logit subtraction removes signal the expert already had right.

**(d) The contrast damages confident predictions most.** In the domain-gap pairing, the per-language loss is largest for high-resource languages where the expert is strongest (en −0.093, fr −0.093) and smallest for lower-resource ones (ar −0.033, es −0.033, ja −0.027). `corr(resource_level, UCD_gain) = −0.58` across pairings — consistent with subtraction eroding the expert's high-confidence answers.

Per-language UCD−Greedy, domain-gap pairing:

| | en | zh | es | fr | de | ar | ko | ja |
|---|---|---|---|---|---|---|---|---|
| greedy | .740 | .693 | .667 | .667 | .640 | .540 | .560 | .633 |
| UCD−greedy | −.093 | −.073 | −.033 | −.093 | −.053 | −.033 | −.060 | −.027 |

---

## 4. Interpretation

Contrastive decoding (Li et al. 2023; O'Brien & Lewis 2023) and UCD (Lee et al. 2025) earn their gains in **open-ended generation**, where contrasting an expert against a weaker base across *many* tokens suppresses repetition and hallucination. **Single-step multiple-choice answer selection is the wrong setting**: the task reduces to the expert's argmax over {A,B,C,D}, and subtracting a second model's logits can only move probability *away* from that argmax. The damage scales with how different the base is — hence the domain gap is worst.

Note this does **not** contradict the paper: its MC numbers (TruthfulQA MC1/2/3) score probability mass over **full answer strings** (multi-token), not a single-letter argmax. That motivates the next experiment.

---

## 5. Caveats

- n = 150/language → per-language SE ≈ 0.038, so individual per-language deltas in the small-margin pairings (`0.5B`, `7B`) are within noise. The **across-the-board direction** (greedy ≥ CD ≈ UCD) and the **domain-gap −0.058** are robust.
- Single-step MC scoring is one of several valid MC protocols; §6 tests the multi-token alternative.
- Apollo2-7B substitutes for Meditron (gated); both are domain-pretrained 7B medical models, but they are not identical.

---

## 6. Generation-mode (multi-token) scoring

To test whether the contrast helps in the regime it was *built* for, we re-ran with **cloze / MC2-style** scoring: each answer choice's full **text** is scored as a continuation of the question (`Question: …\nAnswer:`), token-by-token, with the logit trace accumulated over the answer continuation and the energy weight recomputed at every token. The choice with the highest length-normalized sequence log-prob wins. (`scripts/run_generation.py`.) This is a harder, noisier task than letter-MC — absolute accuracy is lower — so compare **within** each mode, not across.

Paired deltas vs greedy over all 1,200 items, with 95 % bootstrap CIs (`*` = CI excludes 0):

| Pairing | Mode | Greedy | CD | UCD | CD − Greedy | UCD − Greedy |
|---|---|---|---|---|---|---|
| `0.5B` | MC | 0.316 | 0.302 | 0.303 | −0.014 [−.033,+.004] | −0.013 [−.029,+.005] |
| `0.5B` | **GEN** | 0.283 | 0.295 | 0.272 | **+0.012** [−.003,+.026] | −0.012 [−.033,+.008] |
| `7Bexp_7Bbase` | MC | 0.671 | 0.665 | 0.665 | −0.006 [−.015,+.003] | −0.006 [−.015,+.003] |
| `7Bexp_7Bbase` | **GEN** | 0.374 | 0.381 | 0.371 | **+0.007** [−.007,+.022] | −0.003 [−.022,+.016] |
| `apollo2_qwen7b` | MC | 0.642 | 0.583 | 0.584 | −0.059 [−.078,−.039]\* | −0.058 [−.077,−.041]\* |
| `apollo2_qwen7b` | **GEN** | 0.371 | 0.347 | 0.315 | −0.023 [−.046,−.002]\* | −0.056 [−.083,−.030]\* |

**Findings:**

1. **CD's effect flips sign with the scoring mode.** For the two same-family pairings, `CD − Greedy` goes from *negative* in MC to *positive* in generation (`0.5B` −0.014 → +0.012; `7B+7B` −0.006 → +0.007). The flip is consistent in direction across pairings, though each individual delta's CI still includes 0 at n = 150/lang. This is direct evidence that **contrastive decoding is a multi-token / generation-time effect** — exactly where the original CD and UCD papers found their gains.

2. **The domain gap hurts in both modes, but less in generation.** `apollo2_qwen7b` CD−Greedy improves from −0.059 (MC, significant) to −0.023 (GEN, still significant but ~2.5× smaller). Multi-token scoring partially recovers the damage a divergent base does, but doesn't erase it.

3. **UCD's dynamic weighting still doesn't pay off on average** — UCD ≤ CD in every mode/pairing, and UCD−Greedy is never significantly positive. The energy reweighting consistently underperforms the *static* CD it was meant to improve on, for this task.

4. **Where UCD does help, it helps the *high*-resource languages — opposite the hypothesis.** In `7B+7B` generation, UCD beats greedy most on English (+0.040) and French (+0.033) and *hurts* lower-resource German/Arabic/Korean/Japanese (−0.02 to −0.05). The project's premise (bigger UCD gains for lower-resource languages) is contradicted: the contrast helps where the models are already strongest.

7B+7B generation, per-language UCD − Greedy: en **+0.040**, fr **+0.033**, zh +0.007, es +0.007, ja −0.020, ko −0.020, ar −0.027, de −0.047.

**Bottom line:** moving from single-letter MC to multi-token scoring flips CD from mildly harmful to mildly helpful (confirming it's a generation-time mechanism), but UCD's uncertainty-aware weighting adds no reliable benefit over static CD in any setting tested, and the multilingual low-resource hypothesis is not supported. Confirming the CD sign-flip at significance would need larger n (≥ ~500/lang).

---

## 7. Reproduce

```bash
# Each pairing (writes results/records_<tag>_n150.json + summary)
python scripts/run_fast.py --expert Qwen/Qwen2.5-0.5B-Instruct --base Qwen/Qwen2.5-0.5B            --tag 0.5B
python scripts/run_fast.py --expert Qwen/Qwen2.5-7B-Instruct  --base Qwen/Qwen2.5-0.5B            --tag 7Bexp_0.5Bbase
python scripts/run_fast.py --expert Qwen/Qwen2.5-7B-Instruct  --base Qwen/Qwen2.5-7B              --tag 7Bexp_7Bbase
python scripts/run_fast.py --expert FreedomIntelligence/Apollo2-7B --base Qwen/Qwen2.5-7B         --tag apollo2_qwen7b

# β sweep + four-way comparison + figures
python scripts/sweep_beta.py
python scripts/compare_pairings.py 0.5B 7Bexp_0.5Bbase 7Bexp_7Bbase apollo2_qwen7b --n 150
python scripts/generate_plots.py --results_dir results --output_dir plots/<tag> --model_size <tag> --n_samples 150
```

Figures per pairing: `plots/<tag>/fig{1..5}.png`. Engine correctness tests: `python tests/test_ucd_engine.py` (8 passing).
