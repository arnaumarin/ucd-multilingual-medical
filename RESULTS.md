# Results — Does UCD Help Multilingual Medical Multiple-Choice QA?

**TL;DR.** Across four expert/base pairings spanning model **scale**, **capability gap**, and **domain gap**, plain **greedy (expert-only) decoding beats both Contrastive Decoding (CD) and Uncertainty-Aware Contrastive Decoding (UCD)** on MMMLU-Medical single-step multiple-choice across 8 languages. UCD ≈ CD everywhere — the energy-based dynamic weighting adds nothing measurable. Critically, when the contrast is made genuinely "live" (expert/base energies de-correlate), it still doesn't help, and a **true medical domain gap makes accuracy ~6 points *worse***. We attribute this to contrastive decoding being a **generation-time** mechanism that does not transfer to single-token MC answer selection.

A follow-up **generation-mode** experiment (§6, multi-token cloze scoring) was run to test whether the contrast helps in the regime it was designed for. At n = 500/language it does **not**: moving to multi-token scoring *reduces* the contrast's harm (the significant MC penalty becomes non-significant for same-family pairings) but produces **no benefit** — CD−greedy stays ≤ 0, and UCD never beats greedy in any setting. (An apparent CD "flip to helpful" seen in an n = 150 pilot did not survive at n = 500; see §6.)

This is a negative result, reported in full. **All numbers below are at n = 500/language (4,000 paired items per cell)** unless noted.

---

## 1. Setup

| | |
|---|---|
| **Data** | MMMLU-Medical — `openai/MMMLU` for non-English, `cais/mmlu` for English. 9 medical subjects (clinical knowledge, medical genetics, anatomy, professional/college medicine, college/HS biology, nutrition, virology). **500 stratified samples/language × 8 languages = 4,000 items per pairing** (an earlier n = 150 pilot is superseded). |
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

## 2. Main result (single-step MC, n = 4,000/pairing)

Accuracy averaged over the 8 languages. `*` on a delta = 95 % paired-bootstrap CI excludes 0.

| Pairing | Greedy | CD | UCD | UCD−Greedy | corr(E_exp,E_base) | w_exp std | Contrast "live"? |
|---|---|---|---|---|---|---|---|
| `0.5B` | **0.344** | 0.328 | 0.330 | −0.015\* | 0.999 | 0.006 | no (weights collapsed) |
| `7Bexp_0.5Bbase` | **0.661** | 0.661 | 0.661 | −0.001 | 0.992 | 0.022 | **yes** |
| `7Bexp_7Bbase` | **0.661** | 0.652 | 0.653 | −0.009\* | 1.000 | 0.006 | no |
| `apollo2_qwen7b` | **0.645** | 0.581 | 0.583 | **−0.061\*** | 0.989 | 0.022 | **yes (domain)** |

**Greedy wins in every regime, now with significance.** UCD ≡ CD to within ≤0.003 in all four pairings — the dynamic weighting is inert. The contrast *significantly hurts* in three of four pairings (only the large capability gap is a wash), and the **medical domain gap hurts most: −0.061** (CI [−0.071, −0.051]).

---

## 3. Mechanism — why the contrast doesn't help (and the experiment that proves it)

**(a) The dynamic weight collapses when the two models are similar.** UCD weights the expert by `w_exp = E_exp / (E_exp + E_base)`. With β = 1, the cumulative energy is **99% the logit trace** (a sum over all prompt tokens). For similar models the expert/base energies are nearly identical (`corr = 0.999`), so `w_exp = 0.4965 ± 0.006` — a constant ≈ 0.5. UCD degenerates into static CD.

**(b) β doesn't rescue it.** A β-sweep (0.0 → 1.0) on the 0.5B pairing restores weight variance (std 0.006 → 0.034 as β → 0) but **accuracy is flat** (UCD ≈ −0.013 vs greedy at every β). The weighting mechanism was never the bottleneck.

**(c) A "live" contrast still doesn't help — and a domain gap hurts most.** This is the decisive test. In the **capability-gap** (`7Bexp_0.5Bbase`) and **domain-gap** (`apollo2_qwen7b`) pairings the energies genuinely de-correlate (`corr ≈ 0.989`) and the weight variance more than triples (`w_std ≈ 0.022`) — the contrast is fully engaged. Yet:
- the capability gap is a wash (−0.001), and
- the **domain gap hurts the most of any pairing (−0.061, significant)**.

The more the expert and base genuinely differ in *what they know*, the more the logit subtraction removes signal the expert already had right.

**(d) The damage is broad across languages.** In the domain-gap pairing the loss is significant and spread across all 8 languages, largest for French (−0.104) and high-resource English (−0.066). `corr(resource_level, UCD_gain) = −0.32`, i.e. a weak tendency for higher-resource languages to be hurt more (consistent with subtraction eroding the expert's confident answers), but the contrast hurts everywhere.

Per-language UCD−Greedy, domain-gap pairing (n = 500/lang):

| | en | zh | es | fr | de | ar | ko | ja |
|---|---|---|---|---|---|---|---|---|
| greedy | .724 | .696 | .688 | .684 | .646 | .524 | .546 | .652 |
| UCD−greedy | −.066 | −.058 | −.048 | −.104 | −.052 | −.052 | −.064 | −.048 |

---

## 4. Interpretation

Contrastive decoding (Li et al. 2023; O'Brien & Lewis 2023) and UCD (Lee et al. 2025) earn their gains in **open-ended generation**, where contrasting an expert against a weaker base across *many* tokens suppresses repetition and hallucination. **Single-step multiple-choice answer selection is the wrong setting**: the task reduces to the expert's argmax over {A,B,C,D}, and subtracting a second model's logits can only move probability *away* from that argmax. The damage scales with how different the base is — hence the domain gap is worst.

Note this does **not** contradict the paper: its MC numbers (TruthfulQA MC1/2/3) score probability mass over **full answer strings** (multi-token), not a single-letter argmax. §6 tests that multi-token regime directly — and finds the contrast is *less harmful* there, but still not helpful for this task.

---

## 5. Caveats

- All numbers are n = 500/language (4,000 paired items/cell). A pilot at n = 150 over-stated some small effects — notably an apparent CD benefit in generation mode (§6) that vanished at n = 500. Reported deltas marked `*` have 95 % paired-bootstrap CIs excluding 0.
- Single-step MC scoring is one of several valid MC protocols; §6 tests the multi-token alternative.
- Apollo2-7B substitutes for Meditron (gated); both are domain-pretrained 7B medical models, but they are not identical.

---

## 6. Generation-mode (multi-token) scoring

To test whether the contrast helps in the regime it was *built* for, we re-ran with **cloze / MC2-style** scoring: each answer choice's full **text** is scored as a continuation of the question (`Question: …\nAnswer:`), token-by-token, with the logit trace accumulated over the answer continuation and the energy weight recomputed at every token. The choice with the highest length-normalized sequence log-prob wins. (`scripts/run_generation.py`.) This is a harder, noisier task than letter-MC — absolute accuracy is lower — so compare **within** each mode, not across.

Paired deltas vs greedy over all 4,000 items, with 95 % bootstrap CIs (`*` = CI excludes 0):

| Pairing | Mode | Greedy | CD | UCD | CD − Greedy | UCD − Greedy |
|---|---|---|---|---|---|---|
| `0.5B` | MC | 0.344 | 0.328 | 0.330 | −0.016 [−.026,−.006]\* | −0.015 [−.025,−.004]\* |
| `0.5B` | **GEN** | 0.304 | 0.302 | 0.289 | −0.002 [−.010,+.007] | −0.015 [−.025,−.003]\* |
| `7Bexp_7Bbase` | MC | 0.661 | 0.652 | 0.653 | −0.009 [−.014,−.005]\* | −0.009 [−.013,−.004]\* |
| `7Bexp_7Bbase` | **GEN** | 0.386 | 0.380 | 0.387 | −0.006 [−.015,+.003] | +0.001 [−.010,+.012] |
| `apollo2_qwen7b` | MC | 0.645 | 0.581 | 0.584 | −0.064 [−.074,−.053]\* | −0.061 [−.071,−.051]\* |
| `apollo2_qwen7b` | **GEN** | 0.385 | 0.361 | 0.345 | −0.025 [−.037,−.012]\* | −0.040 [−.055,−.026]\* |

**Findings (n = 500):**

1. **There is no CD "flip."** An n = 150 pilot suggested CD − Greedy turned *positive* in generation (`0.5B` +0.012, `7B+7B` +0.007). At n = 500 both collapse to ≈ 0 and slightly negative (−0.002, −0.006; CIs include 0). The pilot effect was sampling noise — caught precisely by raising n. CD does **not** beat greedy in generation mode.

2. **Generation mode reduces the contrast's harm but adds no benefit.** For same-family pairings the *significant* MC penalty (`0.5B` −0.016\*, `7B+7B` −0.009\*) becomes *non-significant* in generation (−0.002, −0.006). So multi-token scoring is less hostile to contrastive decoding than single-letter MC — consistent with CD being a generation-time mechanism — but the net effect is "harmless," not "helpful."

3. **The domain gap hurts significantly in both modes** — `apollo2_qwen7b` CD−Greedy −0.064\* (MC) → −0.025\* (GEN). Multi-token scoring more than halves the damage a divergent base does, but it stays significantly negative.

4. **UCD's dynamic weighting never pays off.** UCD−Greedy is significantly *negative* in 4 of 6 cells and never significantly positive; UCD ≤ CD throughout. The energy reweighting consistently fails to beat the static CD it was meant to improve on.

5. **The multilingual low-resource hypothesis is not supported.** Even the per-language texture that hinted at it in the pilot is weak at n = 500: in `7B+7B` generation, UCD's only above-noise per-language gain is English (+0.030), with low-resource German/Arabic *hurt* (−0.036, −0.022). Where the contrast does least harm, it is on the *highest*-resource language — the opposite of the premise.

**Bottom line:** at adequate sample size, neither CD nor UCD beats greedy in either single-letter MC or multi-token generation. Generation mode is *less harmful* to the contrast (supporting "CD is a generation-time effect"), but not beneficial here; UCD's uncertainty-aware weighting adds no reliable value over static CD in any setting, and the low-resource hypothesis is contradicted.

---

## 7. Reproduce

```bash
# Single-step MC, each pairing (writes results/records_<tag>_n500.json + summary).
# Two-7B pairings need expandable_segments to avoid OOM on long medical vignettes.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/run_fast.py --expert Qwen/Qwen2.5-0.5B-Instruct --base Qwen/Qwen2.5-0.5B      --tag 0.5B           --n 500
python scripts/run_fast.py --expert Qwen/Qwen2.5-7B-Instruct  --base Qwen/Qwen2.5-0.5B      --tag 7Bexp_0.5Bbase --n 500
python scripts/run_fast.py --expert Qwen/Qwen2.5-7B-Instruct  --base Qwen/Qwen2.5-7B        --tag 7Bexp_7Bbase   --n 500
python scripts/run_fast.py --expert FreedomIntelligence/Apollo2-7B --base Qwen/Qwen2.5-7B   --tag apollo2_qwen7b --n 500

# Generation-mode (multi-token cloze) for the three core pairings
python scripts/run_generation.py --expert Qwen/Qwen2.5-0.5B-Instruct --base Qwen/Qwen2.5-0.5B    --tag 0.5B           --n 500
python scripts/run_generation.py --expert Qwen/Qwen2.5-7B-Instruct  --base Qwen/Qwen2.5-7B       --tag 7Bexp_7Bbase   --n 500
python scripts/run_generation.py --expert FreedomIntelligence/Apollo2-7B --base Qwen/Qwen2.5-7B  --tag apollo2_qwen7b --n 500

# β sweep + four-way MC comparison + figures
python scripts/sweep_beta.py
python scripts/compare_pairings.py 0.5B 7Bexp_0.5Bbase 7Bexp_7Bbase apollo2_qwen7b --n 500
python scripts/generate_plots.py --results_dir results --output_dir plots/<tag> --model_size <tag> --n_samples 500
```

Figures per pairing: `plots/<tag>/fig{1..5}.png`. Engine correctness tests: `python tests/test_ucd_engine.py` (8 passing).
