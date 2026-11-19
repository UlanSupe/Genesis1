from fractions import Fraction

import pytest

from conftest import ALL4, finding, new_env, submit
from traceforge.evaluation.metrics import make_metrics, precision_recall, summarize
from traceforge.evaluation.reward import f1_fraction, invalid_reward, terminal_reward


@pytest.mark.parametrize("tp,fp,fn,want", [
    (0, 0, 0, Fraction(1)),            # nothing to find, nothing claimed
    (0, 3, 0, Fraction(0)),            # empty truth, predictions exist
    (0, 0, 2, Fraction(0)),            # missed everything
    (4, 0, 0, Fraction(1)), (1, 1, 0, Fraction(2, 3)), (1, 0, 1, Fraction(2, 3)),
    (2, 1, 1, Fraction(2, 3)), (3, 27, 1, Fraction(6, 34)),
])
def test_f1_formula_and_special_cases(tp, fp, fn, want):
    assert f1_fraction(tp, fp, fn) == want


@pytest.mark.parametrize("f1,steps,want", [
    (Fraction(1), 0, 1.0), (Fraction(1), 10, 0.98), (Fraction(1), 100, 0.8), (Fraction(0), 0, -1.0),
    (Fraction(0), 50, -1.0), (Fraction(1, 2), 0, 0.0), (Fraction(1, 2), 25, -0.05),
    (Fraction(3, 4), 7, 0.5 - 0.014),
])
def test_terminal_reward_formula(f1, steps, want):
    assert terminal_reward(f1, steps) == pytest.approx(want, abs=1e-12)


def test_reward_is_clipped_to_unit_interval():
    assert terminal_reward(Fraction(0), 100) == -1.0
    assert terminal_reward(Fraction(1), 0) == 1.0
    assert all(-1.0 <= terminal_reward(Fraction(a, 10), s) <= 1.0 for a in range(11) for s in range(0, 101, 10))


def test_step_cost_is_exactly_two_thousandths_and_exact():
    assert terminal_reward(Fraction(1), 1) == 1.0 - 0.002 == 0.998
    assert terminal_reward(Fraction(1), 3) == terminal_reward(Fraction(1), 3)


def test_invalid_structure_reward_is_minus_one():
    assert invalid_reward() == -1.0


def test_precision_recall_conventions():
    assert precision_recall(0, 0, 0) == (1.0, 1.0)
    assert precision_recall(0, 0, 2) == (0.0, 0.0)
    assert precision_recall(0, 2, 0) == (0.0, 1.0)
    assert precision_recall(3, 1, 1) == (0.75, 0.75)


def test_metrics_contain_all_official_fields():
    m = make_metrics(2, 1, 1, 0.3, 12, 0.5)
    for k in ("tp", "fp", "fn", "precision", "recall", "f1", "reward", "inspection_steps", "elapsed_time"):
        assert hasattr(m, k)
    assert (m.tp, m.fp, m.fn, m.inspection_steps) == (2, 1, 1, 12) and m.f1 == pytest.approx(2 / 3)


def test_summarize_means_and_sums():
    s = summarize([make_metrics(1, 0, 0, 1.0, 0, 0), make_metrics(0, 1, 1, -1.0, 10, 0)])
    assert s["episodes"] == 2 and s["tp"] == 1 and s["mean_reward"] == 0.0 and s["mean_inspection_steps"] == 5


def test_end_to_end_reward_with_real_episode():
    env, obs, _ = new_env(1, **ALL4)
    truth = env.evaluator_truth()
    for _ in range(10):
        env.step({"action": "calculator", "expression": "1"})
    _, reward, *_ = submit(env, [finding(t) for t in truth[:3]])        # 3 TP, 0 FP, 1 FN -> F1 = 6/7
    assert reward == pytest.approx(2 * 6 / 7 - 1 - 0.002 * 10)
    m = env.result.metrics
    assert (m.tp, m.fp, m.fn, m.inspection_steps) == (3, 0, 1, 10)


def test_reward_identical_for_same_counts_and_steps():
    rewards = []
    for seed in (1, 2, 3):
        env, obs, _ = new_env(seed, **ALL4)
        env.step({"action": "calculator", "expression": "1"})
        submit(env, [finding(t) for t in env.evaluator_truth()])
        rewards.append(env.result.reward)
    assert rewards == [0.998] * 3
