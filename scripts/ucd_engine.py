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
        "ar":    "السؤال: {q}\nأ. {a}\nب. {b}\nج. {c}\nد. {d}\nالإجابة:",
        "ko":    "질문: {q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n답변:",
        "ja":    "質問: {q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n答え:",
    }.get(language, "Question: {q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:")

    return prefix.format(q=question, a=choices[0], b=choices[1], c=choices[2], d=choices[3])


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
        outputs = model(input_ids, output_hidden_states=False)
        all_logits = outputs.logits  # [1, seq_len, vocab_size]

    seq_len = all_logits.shape[1]

    # Build cumulative logit trace over the sequence (eq. 2)
    # l_t = beta * l_{t-1} + z_t[x_t]  where x_t is the actual token at position t
    logit_trace = 0.0
    for k in range(seq_len - 1):
        token_id = input_ids[0, k + 1].item()   # token selected at step k+1
        token_logit = all_logits[0, k, token_id].item()
        logit_trace = beta * logit_trace + token_logit

    final_logits = all_logits[0, -1, :]  # logits at answer position
    return final_logits, logit_trace


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
    alpha: float,
) -> torch.Tensor:
    """
    UCD logit vector (eq. 4 from paper):
      z_UCD[v] = 2 * w_EXP * z_EXP[v] - w_BASE * z_BASE[v]

    Weights are energy-normalized, applied only when both energies > 0.
    """
    total_energy = expert_energy + base_energy
    if total_energy <= 0 or expert_energy <= 0:
        # Fall back to expert-only when uncertain (paper §3.2.2)
        return expert_logits

    w_exp = expert_energy / total_energy
    w_base = base_energy / total_energy
    return 2 * w_exp * expert_logits - w_base * base_logits


def get_answer_token_ids(tokenizer) -> dict[str, int]:
    """
    Get token IDs for answer letters A/B/C/D.
    Handles tokenizers that prepend a space.
    """
    ids = {}
    for letter in ANSWER_CHOICES:
        # Try plain letter first, then space-prefixed (common in LLaMA/Qwen tokenizers)
        tok_id = tokenizer.encode(letter, add_special_tokens=False)
        if len(tok_id) == 1:
            ids[letter] = tok_id[0]
        else:
            tok_id = tokenizer.encode(" " + letter, add_special_tokens=False)
            ids[letter] = tok_id[-1]
    return ids


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

    if method == "greedy":
        choice_logits = {ch: exp_logits[tid].item() for ch, tid in answer_ids.items()}
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
    base_energy = compute_energy(base_logits, base_trace, config.temperature)

    if method == "cd":
        # Standard contrastive decoding: static equal weighting
        cd_logits = 2 * exp_logits - base_logits
        choice_logits = {ch: cd_logits[tid].item() for ch, tid in answer_ids.items()}
    else:  # ucd
        ucd_logits = ucd_score(exp_logits, base_logits, exp_energy, base_energy, config.alpha)
        choice_logits = {ch: ucd_logits[tid].item() for ch, tid in answer_ids.items()}

    predicted = max(choice_logits, key=choice_logits.get)
    return {
        "predicted": predicted,
        "choice_logits": choice_logits,
        "exp_energy": exp_energy,
        "base_energy": base_energy,
    }
