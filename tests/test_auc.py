"""Hint-AUC: the trapezoidal integral over the hint-ratio grid."""
import math

import hintauc


def test_default_grid_is_the_paper_grid():
    assert tuple(hintauc.DEFAULT_ALPHAS) == (0.0, 0.01, 0.03, 0.05, 0.10, 0.25, 0.50, 1.00)


def test_trapz_matches_closed_form():
    assert math.isclose(hintauc.trapz([0.0, 0.5, 1.0], [0.0, 1.0, 2.0]), 1.0)
    assert math.isclose(hintauc.trapz([0.0, 1.0], [3.0, 3.0]), 3.0)


def test_constant_curve_integrates_to_the_constant():
    assert math.isclose(hintauc.hint_auc({a: 0.7 for a in hintauc.DEFAULT_ALPHAS}), 0.7)


def test_hint_auc_is_independent_of_dict_order():
    assert math.isclose(hintauc.hint_auc({1.0: 0.0, 0.0: 1.0, 0.5: 0.5}), 0.5)
    assert math.isclose(hintauc.hint_auc({0.0: 1.0, 0.5: 0.5, 1.0: 0.0}), 0.5)


def test_hint_auc_table_per_metric():
    table = hintauc.hint_auc_table({0.0: {"psnr": 10.0, "lpips": 0.4}, 1.0: {"psnr": 20.0, "lpips": 0.2}})
    assert set(table) == {"psnr", "lpips"}
    assert math.isclose(table["psnr"], 15.0) and math.isclose(table["lpips"], 0.3)
    assert list(hintauc.hint_auc_table({0.0: {"a": 1, "b": 1}, 1.0: {"a": 1, "b": 1}}, metrics=["b"])) == ["b"]


def test_docstring_example_value():
    curve = {0.0: 0.44, 0.01: 0.33, 0.03: 0.27, 0.05: 0.24, 0.10: 0.20, 0.25: 0.15, 0.50: 0.12, 1.00: 0.10}
    value = hintauc.hint_auc(curve)
    assert 0.10 < value < 0.20            # a decreasing curve integrates between its end points
    assert math.isclose(value, hintauc.trapz(sorted(curve), [curve[a] for a in sorted(curve)]))
