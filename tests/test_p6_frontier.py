"""
The arithmetic behind P6's two additions to the .vn blind-spot argument. Run: pytest -q

Both additions make claims a reader cannot check by eye, so the claims live in pure functions and
the functions are pinned here. The budget frontier states an IMPOSSIBILITY, which is only as good
as the optimality of the construction behind it, so it is re-derived the slow, obviously-correct
way and compared.

Why each test exists: docs/design-notes/test-p6-frontier.md
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
try:
    import _path
    _path.add_script_dirs()
except ImportError:  # flat public-mirror layout: scripts/ itself is already on sys.path
    pass
import run_p6_budget_frontier as bf
import run_p6_vn_deficit as vd


# --------------------------------------------------------------------------- ROC staircase lookups
def _staircase():
    """A hand-built ROC: (fpr, tpr) pairs, both non-decreasing, four operating points."""
    fpr = np.array([0.0, 0.1, 0.4, 1.0])
    tpr = np.array([0.0, 0.5, 0.8, 1.0])
    return fpr, tpr


def test_min_fpr_for_miss_picks_the_cheapest_point_that_clears_the_ceiling():
    fpr, tpr = _staircase()
    # miss <= 0.5 means recall >= 0.5, first reached at fpr 0.1
    assert bf.min_fpr_for_miss(fpr, tpr, 0.5) == pytest.approx(0.1)
    # a ceiling BETWEEN two steps must buy the next step up, not the one below it
    assert bf.min_fpr_for_miss(fpr, tpr, 0.3) == pytest.approx(0.4)
    assert bf.min_fpr_for_miss(fpr, tpr, 0.0) == pytest.approx(1.0)
    # a ceiling above what the curve already gives for free costs nothing
    assert bf.min_fpr_for_miss(fpr, tpr, 1.0) == pytest.approx(0.0)


def test_min_miss_for_fpr_never_overspends():
    fpr, tpr = _staircase()
    assert bf.min_miss_for_fpr(fpr, tpr, 0.1) == pytest.approx(0.5)
    # 0.39 does not buy the 0.4 point: spending more than the budget is not an option
    assert bf.min_miss_for_fpr(fpr, tpr, 0.39) == pytest.approx(0.5)
    assert bf.min_miss_for_fpr(fpr, tpr, 0.4) == pytest.approx(0.2)
    assert bf.min_miss_for_fpr(fpr, tpr, 0.0) == pytest.approx(1.0)
    # a negative remainder means the other group already spent the whole budget
    assert bf.min_miss_for_fpr(fpr, tpr, -0.2) == pytest.approx(1.0)


def test_frontier_matches_brute_force():
    """The optimality argument, checked against exhaustive search on a small synthetic corpus."""
    rng = np.random.RandomState(0)
    # two groups with deliberately different score locations, the situation the paper is about
    s_vn_ph = rng.normal(0.2, 0.6, 40)
    s_vn_be = rng.normal(-1.2, 0.6, 120)
    s_ot_ph = rng.normal(1.4, 0.6, 90)
    s_ot_be = rng.normal(0.1, 0.6, 60)
    pi_vn = len(s_vn_be) / (len(s_vn_be) + len(s_ot_be))

    roc_vn = bf.roc_arrays(np.r_[np.ones(len(s_vn_ph)), np.zeros(len(s_vn_be))],
                           np.r_[s_vn_ph, s_vn_be])
    roc_ot = bf.roc_arrays(np.r_[np.ones(len(s_ot_ph)), np.zeros(len(s_ot_be))],
                           np.r_[s_ot_ph, s_ot_be])

    b = 0.25
    taus = np.array([0.05, 0.1, 0.2, 0.4, 0.6, 0.9])
    got = bf.frontier_curve(roc_vn, roc_ot, pi_vn, b, taus)

    # exhaustive: every candidate threshold in each group, scored the slow way
    cand_vn = np.unique(np.r_[s_vn_ph, s_vn_be, np.inf])
    cand_ot = np.unique(np.r_[s_ot_ph, s_ot_be, np.inf])
    stats_vn = [(float((s_vn_ph < t).mean()), float((s_vn_be >= t).mean())) for t in cand_vn]
    stats_ot = [(float((s_ot_ph < t).mean()), float((s_ot_be >= t).mean())) for t in cand_ot]
    for tau, best in zip(taus, got):
        brute = 1.0
        for miss_v, fpr_v in stats_vn:
            if miss_v > tau + 1e-12:
                continue
            for miss_o, fpr_o in stats_ot:
                if pi_vn * fpr_v + (1 - pi_vn) * fpr_o <= b + 1e-12:
                    brute = min(brute, miss_o)
        assert best == pytest.approx(brute, abs=1e-9), f"tau={tau}: {best} vs brute {brute}"


def test_frontier_is_monotone_and_pareto():
    """Relaxing the .vn ceiling can never make the other group's best worse, and the resulting
    curve must contain no dominated point --- the two properties that make it a frontier."""
    rng = np.random.RandomState(1)
    roc_vn = bf.roc_arrays(np.r_[np.ones(50), np.zeros(150)],
                           np.r_[rng.normal(0.3, 0.7, 50), rng.normal(-1.0, 0.7, 150)])
    roc_ot = bf.roc_arrays(np.r_[np.ones(120), np.zeros(80)],
                           np.r_[rng.normal(1.2, 0.7, 120), rng.normal(0.0, 0.7, 80)])
    taus = np.linspace(0, 1, 51)
    curve = bf.frontier_curve(roc_vn, roc_ot, 0.6, 0.25, taus)
    assert np.all(np.diff(curve) <= 1e-12), "miss_other must be non-increasing in the .vn ceiling"
    keep = np.r_[True, np.diff(curve) < -1e-12]     # the strictly-improving steps
    assert bf.is_pareto_frontier(taus[keep], curve[keep])


def test_is_pareto_frontier_rejects_a_dominated_point():
    assert not bf.is_pareto_frontier(np.array([0.1, 0.2]), np.array([0.5, 0.6]))
    assert bf.is_pareto_frontier(np.array([0.1, 0.2]), np.array([0.6, 0.5]))


def test_price_of_parity_is_the_two_groups_costs_weighted_by_the_benign_shares():
    fpr, tpr = _staircase()
    easy = (fpr, tpr)
    # a group whose curve is strictly worse: it needs fpr 0.4 for the ceiling the other buys at 0.1
    hard = (np.array([0.0, 0.4, 1.0]), np.array([0.0, 0.5, 1.0]))
    # both groups on the same curve: the weighted sum collapses to that curve's own cost
    assert bf.min_budget_for_both(easy, easy, 0.61, 0.5) == pytest.approx(0.1)
    # the .vn share is what decides how much the harder group's price hurts the corpus rate
    assert bf.min_budget_for_both(hard, easy, 0.61, 0.5) == pytest.approx(0.61 * 0.4 + 0.39 * 0.1)
    assert bf.min_budget_for_both(easy, hard, 0.61, 0.5) == pytest.approx(0.61 * 0.1 + 0.39 * 0.4)


# ------------------------------------------------------------------------- margin-space arithmetic
def test_threshold_margin_is_a_logit_and_the_default_threshold_sits_at_zero():
    """deficit_i = margin(threshold) - margin_i. At the deployed 0.5 the threshold's margin is
    exactly 0, which is why the paper can say deficit = -margin there; at any other threshold the
    whole distribution shifts by a constant, and that constant is the logit."""
    assert vd.logit(0.5) == pytest.approx(0.0, abs=1e-15)
    assert vd.logit(0.083) == pytest.approx(np.log(0.083 / 0.917))
    assert vd.logit(0.75) == pytest.approx(-vd.logit(0.25))
    # the shift property the secondary operating point relies on
    margin = np.array([-2.0, -0.5, 0.3])
    d05 = vd.logit(0.5) - margin
    d08 = vd.logit(0.083) - margin
    assert np.allclose(d05 - d08, vd.logit(0.5) - vd.logit(0.083))


def test_margin_is_the_inverse_of_the_reported_probability():
    """The identity the script asserts at runtime: a probability-space version of this analysis
    would compare a log-odds SHAP value against a difference of probabilities."""
    for p in (0.001, 0.083, 0.4, 0.904):
        assert 1.0 / (1.0 + np.exp(-vd.logit(p))) == pytest.approx(p)


def test_neutralisation_pins_only_the_named_feature():
    class Echo:
        """Stands in for the model: returns the pinned column, so pinning is observable."""
        def predict(self, X, prediction_type=None):
            return X[:, 1]

    X = np.array([[1.0, 2.0, 3.0], [4.0, 6.0, 5.0]])
    out = vd.neutralised_margin(Echo(), X, 1, 3.0)
    assert np.allclose(out, [3.0, 3.0])
    assert np.allclose(X, [[1.0, 2.0, 3.0], [4.0, 6.0, 5.0]]), "the caller's matrix was mutated"
