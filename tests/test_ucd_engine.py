"""
Numerical correctness tests for the UCD engine, checked against the equations in
Lee et al. (2025). No model download or network access required — synthetic logits
and a tiny stub model are used so this runs in milliseconds.

Run with:  python tests/test_ucd_engine.py     (or:  pytest tests/)
"""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from ucd_engine import (  # noqa: E402
    UCDConfig, compute_energy, compute_logit_trace, ucd_score, align_vocab,
    decide_answer_form, format_mc_prompt, ANSWER_CHOICES,
)

TOL = 1e-4


class StubModel(torch.nn.Module):
    """Minimal model returning fixed logits, for testing compute_logit_trace."""
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))  # gives .parameters() a device
        self._logits = logits

    def forward(self, input_ids, output_hidden_states=False):
        return SimpleNamespace(logits=self._logits)


def test_energy_decomposition():
    """Energy(z, l) == l + E_p[z] + T*H(p)  (Theorem 1 / eq. 5)."""
    torch.manual_seed(0)
    for T in (0.5, 1.0, 2.0):
        z = torch.randn(50) * 3
        trace = 1.7
        energy = compute_energy(z, trace, T)

        p = torch.softmax(z / T, dim=0)
        exp_logit = (p * z).sum().item()
        entropy = -(p * p.clamp_min(1e-12).log()).sum().item()
        expected = trace + exp_logit + T * entropy

        assert abs(energy - expected) < TOL, f"T={T}: {energy} vs {expected}"
    print("PASS  energy decomposition (eq. 3 == eq. 5)")


def test_energy_trace_invariance():
    """Adding a scalar trace shifts energy by exactly that scalar (l is scalar)."""
    z = torch.randn(30)
    base = compute_energy(z, 0.0, 1.0)
    assert abs(compute_energy(z, 5.0, 1.0) - (base + 5.0)) < TOL
    print("PASS  energy is additive in the scalar logit trace")


def test_logit_trace_recursion():
    """compute_logit_trace reproduces the discounted recursion of eq. 2,
    with the most-recent token weighted beta^0 = 1."""
    vocab = 5
    logits = torch.zeros(1, 4, vocab)
    # input_ids = [0,1,2,3]; transitions score logits[k, input_ids[k+1]]
    logits[0, 0, 1] = 2.0   # a
    logits[0, 1, 2] = 4.0   # b
    logits[0, 2, 3] = 8.0   # c  (most recent -> weight beta^0)
    input_ids = torch.tensor([[0, 1, 2, 3]])
    beta = 0.5

    _, trace = compute_logit_trace(StubModel(logits), None, input_ids, beta, 1.0)
    expected = beta**2 * 2.0 + beta * 4.0 + 8.0  # = 10.5
    assert abs(trace - expected) < TOL, f"{trace} vs {expected}"

    # Final logits returned are the answer-position row (not part of the trace).
    final_logits, _ = compute_logit_trace(StubModel(logits), None, input_ids, beta, 1.0)
    assert torch.allclose(final_logits, logits[0, -1, :])
    print("PASS  logit trace recursion + discounting (eq. 2)")


def test_ucd_score_weighting():
    """eq. 4 with alpha=1 gives 2*w_exp*z_exp - w_base*z_base."""
    z_exp = torch.tensor([1.0, 2.0, 3.0])
    z_base = torch.tensor([0.5, 0.5, 0.5])
    Ee, Eb = 6.0, 2.0
    out = ucd_score(z_exp, z_base, Ee, Eb, alpha=1.0)

    w_exp, w_base = Ee / (Ee + Eb), Eb / (Ee + Eb)
    expected = 2 * w_exp * z_exp - w_base * z_base
    assert torch.allclose(out, expected, atol=TOL)
    print("PASS  ucd_score energy-normalized weighting (eq. 4)")


def test_ucd_score_fallback():
    """Combine only when BOTH energies > 0 (paper 3.2.1); else expert-only."""
    z_exp = torch.tensor([1.0, 2.0, 3.0])
    z_base = torch.tensor([9.0, 0.0, 0.0])
    # base energy <= 0 -> fall back to expert (must NOT add base logits)
    assert torch.allclose(ucd_score(z_exp, z_base, 5.0, -1.0, 1.0), z_exp)
    # expert energy <= 0 -> fall back to expert
    assert torch.allclose(ucd_score(z_exp, z_base, -1.0, 5.0, 1.0), z_exp)
    print("PASS  ucd_score fallback requires both energies > 0")


def test_decide_answer_form():
    """Picks whichever surface form (bare/space) the expert favors overall."""
    answer_ids = {L: {"bare": i, "space": i + 4} for i, L in enumerate(ANSWER_CHOICES)}
    logits = torch.zeros(8)
    logits[4:8] = 5.0  # space-prefixed ids 4..7 dominate
    chosen, form = decide_answer_form(logits, answer_ids)
    assert form == "space" and chosen["A"] == 4

    logits = torch.zeros(8)
    logits[0:4] = 5.0  # bare ids 0..3 dominate
    chosen, form = decide_answer_form(logits, answer_ids)
    assert form == "bare" and chosen["A"] == 0
    print("PASS  decide_answer_form selects the dominant surface form")


def test_align_vocab():
    """Different lm_head widths (e.g. 0.5B=151936 vs 7B=152064) are truncated to
    the common width so the contrast subtracts identical token ids."""
    a = torch.randn(152064)
    b = torch.randn(151936)
    a2, b2 = align_vocab(a, b)
    assert a2.shape[0] == b2.shape[0] == 151936
    assert torch.allclose(a2, a[:151936]) and torch.allclose(b2, b)
    # the A/B/C/D answer ids (well below 151936) survive truncation
    assert all(idx < 151936 for idx in (32, 33, 34, 35, 362, 425, 356, 422))
    # equal widths are passed through untouched
    c, d = align_vocab(torch.zeros(10), torch.ones(10))
    assert c.shape[0] == 10 and d.shape[0] == 10
    print("PASS  align_vocab truncates to shared width, keeps answer tokens")


def test_prompts_use_latin_markers():
    """All language templates expose Latin A./B./C./D. markers (gold is Latin)."""
    for lang in ["en", "zh", "es", "fr", "de", "ar", "ko", "ja"]:
        p = format_mc_prompt("Q?", ["w", "x", "y", "z"], lang)
        for marker in ("A.", "B.", "C.", "D."):
            assert marker in p, f"{lang} missing marker {marker}"
    print("PASS  all language prompts use Latin A/B/C/D markers")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} engine tests passed.")
