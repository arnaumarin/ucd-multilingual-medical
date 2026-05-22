"""
Uncertainty-Aware Contrastive Decoding (UCD) engine for multiple-choice evaluation.
Implements the energy-based dynamic weighting from Lee et al. (2025, ACL Findings).

For MC questions we use single-step scoring: compute logits over answer tokens
{A, B, C, D} at the 'Answer:' position, then apply UCD weighting.
The logit trace (cumulative history) is built over question tokens, not generated tokens,
giving a more principled uncertainty signal for MC tasks.
"""

import torch
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional


ANSWER_CHOICES = ["A", "B", "C", "D"]


@dataclass
class UCDConfig:
    beta: float = 1.0       # discount factor for logit trace (from paper §3.2.1)
    temperature: float = 1.0  # T parameter from energy formulation (paper §3.3)
    alpha: float = 1.0      # scaling factor for expert contribution (paper eq. 4)


def format_mc_prompt(question: str, choices: list[str], language: str = "en") -> str:
    """Format a multiple-choice question into a prompt."""
    prefix = {
        "en":    "Question: {q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:",
        "zh":    "问题：{q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n答案：",
        "es":    "Pregunta: {q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nRespuesta:",
        "fr":    "Question : {q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nRéponse :",
        "de":    "Frage: {q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAntwort:",
        # Latin A/B/C/D markers for all languages: MMMLU gold labels are Latin
        # ("A".."D"), and keeping the option markers Latin makes the scored answer
        # token consistent across languages. Only the lead words are translated.
        "ar":    "السؤال: {q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nالإجابة:",
        "ko":    "질문: {q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n답변:",
        "ja":    "質問: {q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n答え:",
    }.get(language, "Question: {q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:")

    return prefix.format(q=question, a=choices[0], b=choices[1], c=choices[2], d=choices[3])


CLOZE_LEAD = {
    "en": "Question: {q}\nAnswer:",
    "zh": "问题：{q}\n答案：",
    "es": "Pregunta: {q}\nRespuesta:",
    "fr": "Question : {q}\nRéponse :",
    "de": "Frage: {q}\nAntwort:",
    "ar": "السؤال: {q}\nالإجابة:",
    "ko": "질문: {q}\n답변:",
    "ja": "質問: {q}\n答え:",
}


def format_cloze_context(question: str, language: str = "en") -> str:
    """Context for generation-mode scoring: the question + an 'Answer:' cue, with
    NO listed options. Each answer choice's full TEXT is then scored as the
    continuation (MC2/MC3-style cloze)."""
    return CLOZE_LEAD.get(language, CLOZE_LEAD["en"]).format(q=question)


def compute_logit_trace(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    beta: float,
    temperature: float,
) -> tuple[torch.Tensor, float]:
    """
    Run forward pass and compute:
    1. Logits at the final (answer) position
    2. Logit trace l_T accumulated over all question tokens (eq. 2 from paper)

    Returns (final_logits, logit_trace_scalar)
    """
    input_ids = input_ids.to(next(model.parameters()).device)
    with torch.no_grad():
        logits = model(input_ids).logits[0]  # [seq_len, vocab] (model dtype)

    # Logit each model assigned to the actually-occurring next token, per position.
    # Gather in model dtype, then cast just this [seq_len-1] vector to fp32 — avoids
    # materializing a full [seq_len, vocab] fp32 copy (OOM on long prompts + 7B pairs).
    sel = logits[:-1].gather(1, input_ids[0, 1:, None]).squeeze(1).float()  # [seq_len-1]

    # Cumulative logit trace (eq. 2): discounted sum with the most recent token
    # weighted beta^0 = 1. Vectorized (no per-token host sync).
    if beta == 1.0:
        logit_trace = float(sel.sum())
    else:
        L = sel.shape[0]
        powers = beta ** torch.arange(L - 1, -1, -1, device=sel.device, dtype=torch.float32)
        logit_trace = float((sel * powers).sum())

    final_logits = logits[-1].float()  # [vocab] at answer position
    return final_logits, logit_trace


def align_vocab(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Align two logit vectors to a common width.

    Qwen2.5 model sizes pad the lm_head differently (e.g. 0.5B -> 151936,
    7B -> 152064) even though the BPE tokenizer/vocab is shared. The extra rows
    are unused padding tokens, so truncating both to min(width) lets the contrast
    operate over identical token ids without losing any real (or answer) token.
    """
    if a.shape[0] != b.shape[0]:
        v = min(a.shape[0], b.shape[0])
        return a[:v], b[:v]
    return a, b


def compute_energy(logits: torch.Tensor, logit_trace: float, temperature: float) -> float:
    """
    Cumulative energy function (eq. 3 from paper):
      Energy(z_t, l_t) = T * log sum_v exp((z_t[v] + l_t) / T)
    """
    scaled = (logits + logit_trace) / temperature
    energy = temperature * torch.logsumexp(scaled, dim=0).item()
    return energy


def ucd_score(
    expert_logits: torch.Tensor,
    base_logits: torch.Tensor,
    expert_energy: float,
    base_energy: float,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    UCD logit vector (eq. 4 from paper):
      z_UCD[v] = (1 + alpha) * w_EXP * z_EXP[v] - w_BASE * z_BASE[v]

    The paper's eq. 4 fixes the expert coefficient at 2, which is the alpha=1
    case here; alpha generalizes the expert/base contrast strength.

    Per paper §3.2.1, models are combined only when BOTH energies are positive
    (a negative energy signals the model is highly uncertain). Otherwise we fall
    back to expert-only — which preserves the expert's argmax for MC scoring.
    """
    if expert_energy <= 0 or base_energy <= 0:
        return expert_logits

    total_energy = expert_energy + base_energy
    w_exp = expert_energy / total_energy
    w_base = base_energy / total_energy
    return (1 + alpha) * w_exp * expert_logits - w_base * base_logits


def get_answer_token_ids(tokenizer) -> dict[str, dict[str, int]]:
    """
    For each answer letter A/B/C/D, return BOTH surface-form token IDs:
      - "bare":  the letter with no leading space (e.g. "A")
      - "space": the letter with a leading space (e.g. " A")

    Which form the model actually emits depends on the prompt: after a half-width
    colon ("Answer:") most tokenizers (LLaMA/Qwen) emit a space-prefixed " A",
    whereas after a full-width CJK colon ("答案：") they emit a bare "A". We keep
    both and let decide_answer_form() pick the right one per prompt.
    """
    ids = {}
    for letter in ANSWER_CHOICES:
        bare = tokenizer.encode(letter, add_special_tokens=False)
        space = tokenizer.encode(" " + letter, add_special_tokens=False)
        ids[letter] = {"bare": bare[-1], "space": space[-1]}
    return ids


def decide_answer_form(expert_logits: torch.Tensor, answer_ids: dict) -> tuple[dict, str]:
    """
    Decide whether the model emits bare or space-prefixed answer letters at this
    position, by comparing the total logit mass the EXPERT assigns to each form
    across A/B/C/D. The chosen token IDs are then used to score ALL methods
    (greedy/CD/UCD), keeping comparisons fair.

    Returns ({letter: token_id}, form_name).
    """
    space_total = sum(float(expert_logits[answer_ids[L]["space"]]) for L in ANSWER_CHOICES)
    bare_total = sum(float(expert_logits[answer_ids[L]["bare"]]) for L in ANSWER_CHOICES)
    form = "space" if space_total >= bare_total else "bare"
    return {L: answer_ids[L][form] for L in ANSWER_CHOICES}, form


def predict_mc(
    expert_model,
    base_model,
    tokenizer,
    prompt: str,
    config: UCDConfig,
    method: str = "ucd",  # "ucd" | "greedy" | "cd"
) -> dict:
    """
    Predict answer for a single MC question using specified method.
    Returns dict with predicted letter and intermediate scores.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)

    answer_ids = get_answer_token_ids(tokenizer)

    # --- Expert model forward pass ---
    exp_logits, exp_trace = compute_logit_trace(
        expert_model, tokenizer, input_ids, config.beta, config.temperature
    )
    exp_energy = compute_energy(exp_logits, exp_trace, config.temperature)

    # Pick the answer surface form (bare vs space-prefixed) from the expert.
    chosen, _ = decide_answer_form(exp_logits, answer_ids)

    if method == "greedy":
        choice_logits = {ch: exp_logits[tid].item() for ch, tid in chosen.items()}
        predicted = max(choice_logits, key=choice_logits.get)
        return {
            "predicted": predicted,
            "choice_logits": choice_logits,
            "exp_energy": exp_energy,
            "base_energy": None,
        }

    # --- Base model forward pass ---
    base_logits, base_trace = compute_logit_trace(
        base_model, tokenizer, input_ids, config.beta, config.temperature
    )
    # Align vocab widths (different model sizes pad the lm_head differently),
    # then recompute the expert energy over the shared vocab for comparability.
    exp_logits, base_logits = align_vocab(exp_logits, base_logits)
    exp_energy = compute_energy(exp_logits, exp_trace, config.temperature)
    base_energy = compute_energy(base_logits, base_trace, config.temperature)

    if method == "cd":
        # Standard contrastive decoding: static equal weighting
        cd_logits = 2 * exp_logits - base_logits
        choice_logits = {ch: cd_logits[tid].item() for ch, tid in chosen.items()}
    else:  # ucd
        ucd_logits = ucd_score(exp_logits, base_logits, exp_energy, base_energy, config.alpha)
        choice_logits = {ch: ucd_logits[tid].item() for ch, tid in chosen.items()}

    predicted = max(choice_logits, key=choice_logits.get)
    return {
        "predicted": predicted,
        "choice_logits": choice_logits,
        "exp_energy": exp_energy,
        "base_energy": base_energy,
    }


def evaluate_mc_sample(
    expert_model,
    base_model,
    tokenizer,
    prompt: str,
    gold: str,
    config: UCDConfig,
    answer_ids: dict,
) -> dict:
    """
    Evaluate greedy / CD / UCD on a single MC question in ONE pair of forward
    passes (one expert, one base). All three methods are scored on the same
    answer tokens, chosen per-prompt via decide_answer_form().

    This is the shared core used by both experiment runners so the engine and
    runners can never drift apart.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    exp_logits, exp_trace = compute_logit_trace(
        expert_model, tokenizer, input_ids, config.beta, config.temperature)
    base_logits, base_trace = compute_logit_trace(
        base_model, tokenizer, input_ids, config.beta, config.temperature)

    # Align vocab widths so the contrast operates over identical token ids
    # (different model sizes pad the lm_head differently).
    exp_logits, base_logits = align_vocab(exp_logits, base_logits)

    exp_energy = compute_energy(exp_logits, exp_trace, config.temperature)
    base_energy = compute_energy(base_logits, base_trace, config.temperature)

    chosen, form = decide_answer_form(exp_logits, answer_ids)

    ucd_logits = ucd_score(exp_logits, base_logits, exp_energy, base_energy, config.alpha)
    cd_logits = 2 * exp_logits - base_logits  # static-weight contrastive decoding

    def pick(vec):
        scores = {ch: vec[tid].item() for ch, tid in chosen.items()}
        return max(scores, key=scores.get), scores

    greedy_pred, greedy_scores = pick(exp_logits)
    cd_pred, cd_scores = pick(cd_logits)
    ucd_pred, ucd_scores = pick(ucd_logits)

    return {
        "gold":          gold,
        "greedy_pred":   greedy_pred,
        "cd_pred":       cd_pred,
        "ucd_pred":      ucd_pred,
        "greedy_ok":     greedy_pred == gold,
        "cd_ok":         cd_pred == gold,
        "ucd_ok":        ucd_pred == gold,
        "exp_energy":    exp_energy,
        "base_energy":   base_energy,
        "exp_trace":     float(exp_trace),
        "base_trace":    float(base_trace),
        "answer_form":   form,
        "greedy_scores": greedy_scores,
        "cd_scores":     cd_scores,
        "ucd_scores":    ucd_scores,
    }


# ── Generation-mode (multi-token / cloze) scoring ────────────────────────────
def _seq_logprobs(model, tokenizer, ctx_ids, cont_ids):
    """Forward `ctx_ids + cont_ids` once; return the fp32 logit rows that predict
    each continuation token. Uses logits_to_keep so the lm_head only projects the
    last few positions to vocab (avoids a full [L, vocab] tensor — OOM on 7B pairs
    with long medical contexts)."""
    full = torch.tensor([ctx_ids + cont_ids], device=next(model.parameters()).device)
    k = len(cont_ids) + 1  # last k positions = (n_ctx-1) .. (total-1)
    with torch.no_grad():
        logits = model(full, logits_to_keep=k).logits[0]   # [k, V] (model dtype)
    # of those, positions 0..len(cont)-1 predict cont_ids[0..]; the last is unused
    return logits[: len(cont_ids)].float()                 # [len(cont), V]


def score_answer_text_generation(
    expert_model, base_model, tokenizer, context: str, choices: list[str],
    config: UCDConfig,
) -> tuple[dict, dict]:
    """
    Generation-mode scoring: score each answer-choice TEXT as a multi-token
    continuation of `context`, under greedy / CD / UCD. This is the regime CD/UCD
    were designed for — the logit trace accumulates over the continuation tokens
    and the energy-based weight is recomputed at every token.

    For each choice we sum per-token log-probabilities (length-normalized) of the
    choice text, then pick the choice with the highest mean log-prob per method.

    Returns ({method: predicted_index}, {method: [per-choice mean logprob]}).
    """
    ctx_ids = tokenizer.encode(context, add_special_tokens=True)
    methods = ["greedy", "cd", "ucd"]
    scores = {m: [] for m in methods}

    for choice in choices:
        cont_ids = tokenizer.encode(" " + choice, add_special_tokens=False)
        if not cont_ids:
            for m in methods:
                scores[m].append(-1e9)
            continue

        exp_rows = _seq_logprobs(expert_model, tokenizer, ctx_ids, cont_ids)
        base_rows = _seq_logprobs(base_model, tokenizer, ctx_ids, cont_ids)

        lp = {m: 0.0 for m in methods}
        trace_e = 0.0
        trace_b = 0.0
        T = config.temperature
        for i, tok in enumerate(cont_ids):
            ze, zb = align_vocab(exp_rows[i], base_rows[i])

            lp["greedy"] += torch.log_softmax(ze, dim=0)[tok].item()
            lp["cd"] += torch.log_softmax(2 * ze - zb, dim=0)[tok].item()

            # per-token energy with the trace over the continuation so far
            Ee = T * torch.logsumexp((ze + trace_e) / T, dim=0).item()
            Eb = T * torch.logsumexp((zb + trace_b) / T, dim=0).item()
            ucd_logits = ucd_score(ze, zb, Ee, Eb, config.alpha)
            lp["ucd"] += torch.log_softmax(ucd_logits, dim=0)[tok].item()

            # advance traces with the logit each model assigned to the chosen token
            trace_e = config.beta * trace_e + ze[tok].item()
            trace_b = config.beta * trace_b + zb[tok].item()

        n = len(cont_ids)
        for m in methods:
            scores[m].append(lp[m] / n)   # length-normalized mean log-prob

    preds = {m: int(max(range(len(choices)), key=lambda j: scores[m][j])) for m in methods}
    return preds, scores


# ── TruthfulQA-style multi-token candidate scoring (paper's MC1/MC2/MC3) ──────
def candidate_logprobs(
    expert_model, base_model, tokenizer, context: str, candidates: list[str],
    config: UCDConfig,
) -> dict:
    """Total (summed) log-prob of each candidate answer string as a continuation of
    `context`, under greedy / CD / UCD — the multi-token, generation-time scoring the
    UCD paper uses for TruthfulQA MC1/MC2/MC3. The logit trace accumulates over the
    candidate's tokens and the energy weight is recomputed per token.

    Returns {method: [sum_logprob per candidate]}.
    """
    ctx_ids = tokenizer.encode(context, add_special_tokens=True)
    methods = ["greedy", "cd", "ucd"]
    out = {m: [] for m in methods}
    T = config.temperature

    for cand in candidates:
        cont_ids = tokenizer.encode(" " + cand.strip(), add_special_tokens=False)
        if not cont_ids:
            for m in methods:
                out[m].append(-1e30)
            continue
        exp_rows = _seq_logprobs(expert_model, tokenizer, ctx_ids, cont_ids)
        base_rows = _seq_logprobs(base_model, tokenizer, ctx_ids, cont_ids)

        lp = {m: 0.0 for m in methods}
        trace_e = 0.0
        trace_b = 0.0
        for i, tok in enumerate(cont_ids):
            ze, zb = align_vocab(exp_rows[i], base_rows[i])
            lp["greedy"] += torch.log_softmax(ze, dim=0)[tok].item()
            lp["cd"] += torch.log_softmax(2 * ze - zb, dim=0)[tok].item()
            Ee = T * torch.logsumexp((ze + trace_e) / T, dim=0).item()
            Eb = T * torch.logsumexp((zb + trace_b) / T, dim=0).item()
            lp["ucd"] += torch.log_softmax(ucd_score(ze, zb, Ee, Eb, config.alpha), dim=0)[tok].item()
            trace_e = config.beta * trace_e + ze[tok].item()
            trace_b = config.beta * trace_b + zb[tok].item()
        for m in methods:
            out[m].append(lp[m])
    return out


def truthfulqa_mc_scores(logprobs: list[float], labels: list[int]) -> tuple[float, float, float]:
    """Compute (MC1, MC2, MC3) for one question from per-candidate total log-probs.
    labels[i] == 1 for true answers, 0 for false. (For MC1 the canonical set has a
    single true answer; MC2/MC3 use the multi-true set.)

    MC1: 1.0 if the highest-scoring candidate is a true answer.
    MC2: normalized probability mass on the true answers.
    MC3: fraction of true answers that outrank every false answer.
    """
    import numpy as np
    s = np.asarray(logprobs, dtype=np.float64)
    lab = np.asarray(labels)
    true_idx = np.where(lab == 1)[0]
    false_idx = np.where(lab == 0)[0]
    if len(true_idx) == 0 or len(false_idx) == 0:
        return float("nan"), float("nan"), float("nan")

    mc1 = 1.0 if lab[int(np.argmax(s))] == 1 else 0.0

    m = s.max()
    probs = np.exp(s - m)
    pt, pf = probs[true_idx].sum(), probs[false_idx].sum()
    mc2 = float(pt / (pt + pf))

    max_false = s[false_idx].max()
    mc3 = float(np.mean(s[true_idx] > max_false))
    return mc1, mc2, mc3
