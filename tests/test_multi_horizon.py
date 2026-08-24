"""Recursive multi-horizon forecasting tests (US-40, PRD §16, §17, §26, §28, §32, §55).

Small hand-built panels isolate one rule each, and the model is always a *baseline* — a rule whose
output can be computed by hand (B1 reads ``lag_1``, B2 reads ``rolling_mean_3``). That is what makes
the recursion checkable: with B1, horizon 2 must return exactly what horizon 1 returned, because
horizon 1's forecast becomes the month that horizon 2 reads as ``lag_1``. A test that could only say
"some number came out" would not prove the feedback loop is wired at all.

Real configuration throughout (``load_model_config()`` / ``load_inventory_policy()``), never a
hand-rolled threshold, so a config change is felt here rather than silently diverging.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pipeline import paths
from pipeline.config import load_cleaning_config, load_inventory_policy, load_model_config
from pipeline.features import build_features, build_features_for_origin
from pipeline.inventory import POLICY_FORECAST_PLUS_SS, target_inventory
from pipeline.latest_forecast import BaselineForecaster
from pipeline.multi_horizon import (
    MONTHS_PER_QUARTER,
    MULTI_HORIZON_COLUMNS,
    PERIOD_MONTH,
    PERIOD_PLAN_COLUMNS,
    PERIOD_QUARTER,
    SOURCE_FORECAST,
    SOURCE_HOLDOUT,
    SOURCE_MIXED,
    build_multi_horizon_plan,
    build_period_plan,
    ensure_month,
    forecast_month_rows,
    holdout_month_rows,
    horizon_months,
    horizon_residuals,
    partial_month_unreachable,
    quarter_of,
    recursive_forecast,
    set_forecast_units,
    validate_multi_horizon,
)
from pipeline.quarterly import quarter_label

CFG = load_model_config()
POLICY = load_inventory_policy()
CLEANING = load_cleaning_config()

B1 = "B1_last_month"
B2 = "B2_ma3"
ORIGIN = "2011-11"
MONTHS = [str(period) for period in pd.period_range("2010-12", "2011-11", freq="M")]


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
def _panel_row(stock_code: str, month: str, units: float) -> dict:
    """One US-05 panel row; only the columns the features read carry meaningful numbers."""
    return {
        "month": month,
        "stock_code": stock_code,
        "description": f"{stock_code} DESCRIPTION",
        "units_sold": units,
        "gross_revenue": units * 2.0,
        "avg_unit_price": 2.0,
        "invoice_count": 1 if units else 0,
        "sale_line_count": 1 if units else 0,
        "customer_count": 1 if units else 0,
        "max_line_qty": units,
        "returned_units": 0,
        "is_partial_month": month in CLEANING.raw.partial_months,
    }


def _panel(units_by_code: dict[str, list[float]], months: list[str] | None = None) -> pd.DataFrame:
    """A contiguous zero-filled panel — one row per product per month, first row a real sale."""
    grid = months if months is not None else MONTHS
    rows = [
        _panel_row(code, month, units)
        for code, series in units_by_code.items()
        for month, units in zip(grid, series, strict=True)
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def steady_panel() -> pd.DataFrame:
    """Two products selling every month — always active under the §14 rule at k = 6."""
    return _panel({"AAA": [10.0] * 11 + [20.0], "BBB": [5.0] * 12})


def _features(panel: pd.DataFrame) -> pd.DataFrame:
    """The training features for every target the panel can support.

    ``horizon_residuals`` reads this frame's first target month to know which back-test origins are
    usable at all — ``lag_3`` needs three months of history, so on a twelve-month test panel the
    configured window's early origins have no feature row. Passing the real frame is what the
    pipeline does; hand-building one would let the test disagree with the feature builder.
    """
    months = sorted(panel["month"].astype(str).unique())
    return build_features(panel, CFG.active_rule.k, months[3], months[-1], CFG)


# --------------------------------------------------------------------------
# calendar helpers
# --------------------------------------------------------------------------
def test_horizon_months_counts_forward_from_the_origin():
    assert horizon_months("2011-11", 3) == ["2011-12", "2012-01", "2012-02"]


def test_horizon_months_rejects_a_zero_horizon():
    with pytest.raises(ValueError, match="at least 1"):
        horizon_months("2011-11", 0)


def test_quarter_of_maps_calendar_quarters():
    assert quarter_of("2011-12") == "2011-Q4"
    assert quarter_of("2012-01") == "2012-Q1"
    assert quarter_of("2012-03") == "2012-Q1"
    # Same spelling as quarterly_forecast.csv, so the two quarterly views can be joined.
    assert quarter_of("2011-08") == quarter_label("2011-08")


# --------------------------------------------------------------------------
# panel amendment
# --------------------------------------------------------------------------
def test_ensure_month_adds_one_row_per_product_and_is_idempotent(steady_panel):
    extended = ensure_month(steady_panel, "2011-12")
    added = extended.loc[extended["month"] == "2011-12"]

    assert len(added) == steady_panel["stock_code"].nunique()
    assert (added["units_sold"] == 0.0).all()
    assert not added["is_partial_month"].any()
    # Already present -> unchanged frame, so a second horizon cannot duplicate a month.
    assert ensure_month(extended, "2011-12").equals(extended)


def test_ensure_month_refuses_to_skip_a_month(steady_panel):
    with pytest.raises(ValueError, match="no rows at"):
        ensure_month(steady_panel, "2012-03")


def test_set_forecast_units_replaces_the_month_wholesale(steady_panel):
    extended = ensure_month(steady_panel, "2011-12")
    forecasts = pd.Series({"AAA": 42.5})  # BBB deliberately absent: inactive at that month

    amended = set_forecast_units(
        extended, "2011-12", forecasts, CFG.multi_horizon.carry_forward_columns
    )
    december = amended.loc[amended["month"] == "2011-12"].set_index("stock_code")

    assert december.loc["AAA", "units_sold"] == pytest.approx(42.5)
    # A product with no forecast is zero, never its recorded actual.
    assert december.loc["BBB", "units_sold"] == 0.0
    # Carried, not invented: the previous month's value, for the two lagged §17 features.
    for column in CFG.multi_horizon.carry_forward_columns:
        assert december.loc["AAA", column] == pytest.approx(
            amended.loc[
                (amended["month"] == "2011-11") & (amended["stock_code"] == "AAA"), column
            ].iloc[0]
        )


# --------------------------------------------------------------------------
# the recursion — the forecast really does become the next month's input
# --------------------------------------------------------------------------
def test_b1_recursion_repeats_its_own_forecast(steady_panel):
    """B1 predicts ``lag_1``. Horizon 1 forecasts November's units; horizon 2 must then forecast
    *that forecast*, because it is what the amended panel holds at December."""
    forecasts = recursive_forecast(
        steady_panel, BaselineForecaster(B1), CFG, ORIGIN, 3, champion=B1
    )
    aaa = forecasts.loc[forecasts["stock_code"] == "AAA"].set_index("horizon")["forecast"]

    assert aaa.loc[1] == pytest.approx(20.0)  # November's units
    assert aaa.loc[2] == pytest.approx(20.0)  # horizon 1's forecast, fed back
    assert aaa.loc[3] == pytest.approx(20.0)


def test_b2_recursion_rolls_the_forecast_into_the_average(steady_panel):
    """B2 predicts ``rolling_mean_3``: the mean of the three months before the target. Horizon 2's
    window is September, October and *horizon 1's forecast* — arithmetic, computed by hand."""
    forecasts = recursive_forecast(
        steady_panel, BaselineForecaster(B2), CFG, ORIGIN, 2, champion=B2
    )
    aaa = forecasts.loc[forecasts["stock_code"] == "AAA"].set_index("horizon")["forecast"]

    h1 = (10.0 + 10.0 + 20.0) / 3  # 2011-09, 2011-10, 2011-11
    h2 = (10.0 + 20.0 + h1) / 3  # 2011-10, 2011-11, and h1 standing in for 2011-12
    assert aaa.loc[1] == pytest.approx(h1)
    assert aaa.loc[2] == pytest.approx(h2)


def test_recursion_records_both_origins(steady_panel):
    """``forecast_origin`` stays the real one; ``step_origin`` names the month the features came
    from — which for horizon 2 is itself a forecast."""
    forecasts = recursive_forecast(
        steady_panel, BaselineForecaster(B1), CFG, ORIGIN, 3, champion=B1
    )

    assert set(forecasts["forecast_origin"]) == {ORIGIN}
    by_horizon = forecasts.drop_duplicates("horizon").set_index("horizon")
    assert by_horizon.loc[2, "step_origin"] == "2011-12"
    assert by_horizon.loc[2, "target_month"] == "2012-01"
    assert by_horizon.loc[3, "target_month"] == "2012-02"


def test_forecasts_are_clipped_at_zero(steady_panel):
    """A negative prediction is not a demand forecast — §19's clip, as in the back-test."""

    class NegativeModel:
        def predict(self, features: pd.DataFrame) -> np.ndarray:
            return np.full(len(features), -5.0)

    forecasts = recursive_forecast(steady_panel, NegativeModel(), CFG, ORIGIN, 2, champion=B1)
    assert (forecasts["forecast"] >= 0).all()


# --------------------------------------------------------------------------
# the partial month (§2.5, §8) — proved by perturbation, not asserted
# --------------------------------------------------------------------------
def test_partial_month_actuals_never_reach_a_horizon():
    """December 2011 is partial. Corrupt its measurements: every horizon must be unchanged,
    because the recursion overwrote that month with its own forecast before reading it."""
    months = [str(period) for period in pd.period_range("2010-12", "2011-12", freq="M")]
    panel = _panel({"AAA": [10.0] * 12 + [3.0], "BBB": [5.0] * 13}, months=months)
    assert panel.loc[panel["month"] == "2011-12", "is_partial_month"].all()

    assert partial_month_unreachable(
        panel, BaselineForecaster(B1), CFG, ORIGIN, 3, CLEANING, champion=B1
    )


def test_the_substitution_is_what_hides_the_partial_month():
    """The guard passes for a reason, and this is the reason.

    Horizon 2 builds its features at origin 2011-12 — the partial month. Read straight from the
    panel, that month's ``units_sold`` is nine days of real sales and it lands in ``lag_1``; after
    :func:`set_forecast_units` has replaced it with horizon 1's forecast, it does not. Without the
    substitution ``partial_month_not_used`` would fail, which is exactly what makes the perturbation
    check a live guard rather than a tautology.
    """
    months = [str(period) for period in pd.period_range("2010-12", "2011-12", freq="M")]
    panel = _panel({"AAA": [10.0] * 12 + [3.0]}, months=months)

    # Horizon 2's target month has to exist before it can have a feature row (ensure_month), and
    # the target itself is 2012-01 — the origin whose lag_1 is the partial December.
    extended = ensure_month(panel, "2012-01")
    raw_features = build_features_for_origin(extended, "2011-12", CFG.active_rule.k, CFG)
    amended = set_forecast_units(
        extended, "2011-12", pd.Series({"AAA": 20.0}), CFG.multi_horizon.carry_forward_columns
    )
    amended_features = build_features_for_origin(amended, "2011-12", CFG.active_rule.k, CFG)

    assert raw_features["lag_1"].iloc[0] == pytest.approx(3.0)  # the partial actual
    assert amended_features["lag_1"].iloc[0] == pytest.approx(20.0)  # horizon 1's forecast


# --------------------------------------------------------------------------
# horizon-specific residuals (§26)
# --------------------------------------------------------------------------
def test_horizon_residuals_never_score_past_the_last_full_month(steady_panel):
    features = _features(steady_panel)
    residuals = horizon_residuals(steady_panel, features, CFG, B1, "2011-11", 2)

    assert not residuals.empty
    assert (residuals["target_month"] <= "2011-11").all()
    assert set(residuals["horizon"]) <= {1, 2}


def test_horizon_residual_is_actual_minus_forecast(steady_panel):
    features = _features(steady_panel)
    residuals = horizon_residuals(steady_panel, features, CFG, B1, "2011-11", 1)

    recomputed = residuals["actual"].astype(float) - residuals["forecast"].astype(float)
    assert np.allclose(residuals["residual"].astype(float), recomputed)


# --------------------------------------------------------------------------
# the plan: forecast + σ -> Recommended Target Inventory (§25, §28)
# --------------------------------------------------------------------------
@pytest.fixture
def plan_inputs(steady_panel):
    forecasts = recursive_forecast(
        steady_panel, BaselineForecaster(B1), CFG, ORIGIN, 2, champion=B1
    )
    features = _features(steady_panel)
    residuals = horizon_residuals(steady_panel, features, CFG, B1, "2011-11", 2)
    abc_train = pd.DataFrame(
        {"stock_code": ["AAA", "BBB"], "abc_class": ["A", "C"]}
    )
    return forecasts, residuals, abc_train, steady_panel


def test_plan_uses_the_section_28_formula(plan_inputs):
    forecasts, residuals, abc_train, panel = plan_inputs
    plan = build_multi_horizon_plan(
        forecasts, residuals, abc_train, panel, POLICY, champion=B1, run_id="test-run"
    )

    assert list(plan.columns) == MULTI_HORIZON_COLUMNS
    expected = target_inventory(
        plan["forecast"].astype(float), plan["sigma"].astype(float), POLICY.z
    )
    assert np.array_equal(plan["target_inventory"].astype(int).to_numpy(), np.asarray(expected))
    assert (plan["z"] == POLICY.z).all()
    assert (plan["safety_stock"] == POLICY.z * plan["sigma"]).all()


def test_plan_covers_every_horizon_and_carries_the_run_id(plan_inputs):
    forecasts, residuals, abc_train, panel = plan_inputs
    plan = build_multi_horizon_plan(
        forecasts, residuals, abc_train, panel, POLICY, champion=B1, run_id="test-run"
    )

    assert sorted(plan["horizon"].unique()) == [1, 2]
    assert (plan["run_id"] == "test-run").all()
    assert plan["sigma"].notna().all()
    assert plan["target_inventory"].notna().all()


# --------------------------------------------------------------------------
# the period view — months, and quarters as the sum of their months
# --------------------------------------------------------------------------
def _month_row(code: str, period: str, forecast: float, ss: float, target: int, source: str,
               actual: float | None) -> dict:
    return {
        "stock_code": code,
        "abc_class": "A",
        "period": period,
        "model": B1,
        "forecast": forecast,
        "safety_stock": ss,
        "target_inventory": target,
        "actual": np.nan if actual is None else actual,
        "source": source,
    }


def test_quarter_row_is_the_sum_of_its_monthly_targets(steady_panel):
    months = pd.DataFrame(
        [
            _month_row("AAA", "2011-10", 10.0, 2.0, 12, SOURCE_HOLDOUT, 9.0),
            _month_row("AAA", "2011-11", 20.0, 3.0, 23, SOURCE_HOLDOUT, 21.0),
            _month_row("AAA", "2011-12", 20.0, 4.0, 24, SOURCE_FORECAST, None),
        ]
    )
    period_plan = build_period_plan(months, steady_panel, run_id="test-run")

    assert list(period_plan.columns) == PERIOD_PLAN_COLUMNS
    quarter = period_plan.loc[period_plan["period_type"] == PERIOD_QUARTER].iloc[0]
    assert quarter["period"] == quarter_of("2011-10")
    assert quarter["target_inventory"] == 12 + 23 + 24
    assert quarter["forecast"] == pytest.approx(50.0)
    assert quarter["safety_stock"] == pytest.approx(9.0)
    assert quarter["n_months"] == MONTHS_PER_QUARTER
    assert bool(quarter["complete"]) is True
    assert quarter["months_included"] == "2011-10;2011-11;2011-12"
    # One month is a forecast, so the quarter reports no actual at all.
    assert pd.isna(quarter["actual"])
    # Two stages contributed, and the row says so rather than claiming either one.
    assert quarter["source"] == SOURCE_MIXED
    # ABC is a per-product property: it survives the rollup rather than being aggregated away.
    assert quarter["abc_class"] == "A"


def test_partial_quarter_is_flagged_not_silently_summed(steady_panel):
    months = pd.DataFrame(
        [
            _month_row("AAA", "2012-01", 10.0, 2.0, 12, SOURCE_FORECAST, None),
            _month_row("AAA", "2012-02", 11.0, 2.0, 13, SOURCE_FORECAST, None),
        ]
    )
    period_plan = build_period_plan(months, steady_panel, run_id="test-run")
    quarter = period_plan.loc[period_plan["period_type"] == PERIOD_QUARTER].iloc[0]

    assert bool(quarter["complete"]) is False
    assert quarter["n_months"] == 2
    assert quarter["months_included"] == "2012-01;2012-02"
    assert quarter["source"] == SOURCE_FORECAST


def test_month_rows_survive_unchanged(steady_panel):
    months = pd.DataFrame(
        [_month_row("AAA", "2011-10", 10.0, 2.0, 12, SOURCE_HOLDOUT, 9.0)]
    )
    period_plan = build_period_plan(months, steady_panel, run_id="test-run")
    month = period_plan.loc[period_plan["period_type"] == PERIOD_MONTH].iloc[0]

    assert month["period"] == "2011-10"
    assert month["target_inventory"] == 12
    assert month["actual"] == pytest.approx(9.0)
    assert month["n_months"] == 1
    assert bool(month["complete"]) is True


def test_holdout_rows_take_only_the_safety_stock_policy_at_the_configured_z():
    sim_rows = pd.DataFrame(
        [
            {
                "stock_code": "AAA",
                "abc_class": "A",
                "target_month": "2011-10",
                "model": B1,
                "policy": policy,
                "z": z,
                "actual": 9.0,
                "forecast": 10.0,
                "sigma": 2.0,
                "target_inventory": 14,
            }
            for policy in ("forecast_only", POLICY_FORECAST_PLUS_SS)
            for z in (POLICY.z, POLICY.z + 0.5)
        ]
    )
    rows = holdout_month_rows(sim_rows, POLICY, champion=B1)

    assert len(rows) == 1
    assert rows["source"].unique().tolist() == [SOURCE_HOLDOUT]
    assert rows["abc_class"].iloc[0] == "A"
    assert rows["safety_stock"].iloc[0] == pytest.approx(POLICY.z * 2.0)


def test_forecast_rows_carry_no_actual(plan_inputs):
    forecasts, residuals, abc_train, panel = plan_inputs
    plan = build_multi_horizon_plan(
        forecasts, residuals, abc_train, panel, POLICY, champion=B1, run_id="test-run"
    )
    rows = forecast_month_rows(plan)

    assert rows["actual"].isna().all()
    assert rows["source"].unique().tolist() == [SOURCE_FORECAST]


# --------------------------------------------------------------------------
# validation (§39 — the guard has to be able to fail)
# --------------------------------------------------------------------------
def _valid_plan() -> pd.DataFrame:
    forecast, sigma = 10.0, 2.0
    return pd.DataFrame(
        [
            {
                "stock_code": "AAA",
                "description": "AAA DESCRIPTION",
                "forecast_origin": ORIGIN,
                "horizon": 1,
                "target_month": "2011-12",
                "model": B1,
                "forecast": forecast,
                "sigma": sigma,
                "sigma_source": "product",
                "n_residuals_product": 12,
                "z": POLICY.z,
                "safety_stock": POLICY.z * sigma,
                "target_inventory": math.ceil(forecast + POLICY.z * sigma),
                "abc_class": "A",
                "status": "Forecast",
                "run_id": "test-run",
            }
        ]
    )


def test_validation_passes_on_a_consistent_plan():
    result = validate_multi_horizon(
        _valid_plan(),
        pd.DataFrame(columns=["stock_code", "target_month", "horizon", "residual"]),
        pd.DataFrame(columns=PERIOD_PLAN_COLUMNS),
        POLICY,
        origin=ORIGIN,
        last_full_month="2011-11",
        partial_month_clean=True,
    )
    assert result.passed, result.summary()


def test_validation_fails_when_a_horizon_targets_the_wrong_month():
    plan = _valid_plan()
    plan.loc[0, "target_month"] = "2012-05"

    result = validate_multi_horizon(
        plan,
        pd.DataFrame(columns=["stock_code", "target_month", "horizon", "residual"]),
        pd.DataFrame(columns=PERIOD_PLAN_COLUMNS),
        POLICY,
        origin=ORIGIN,
        last_full_month="2011-11",
        partial_month_clean=True,
    )
    assert not result.passed
    assert {v.rule for v in result.violations} == {"horizon_target_month"}


def test_validation_fails_when_the_target_inventory_was_not_the_formula():
    plan = _valid_plan()
    plan.loc[0, "target_inventory"] = 999

    result = validate_multi_horizon(
        plan,
        pd.DataFrame(columns=["stock_code", "target_month", "horizon", "residual"]),
        pd.DataFrame(columns=PERIOD_PLAN_COLUMNS),
        POLICY,
        origin=ORIGIN,
        last_full_month="2011-11",
        partial_month_clean=True,
    )
    assert not result.passed
    assert "target_inventory_formula" in {v.rule for v in result.violations}


def test_validation_fails_when_the_partial_month_leaked():
    result = validate_multi_horizon(
        _valid_plan(),
        pd.DataFrame(columns=["stock_code", "target_month", "horizon", "residual"]),
        pd.DataFrame(columns=PERIOD_PLAN_COLUMNS),
        POLICY,
        origin=ORIGIN,
        last_full_month="2011-11",
        partial_month_clean=False,
    )
    assert not result.passed
    assert "partial_month_not_used" in {v.rule for v in result.violations}


def test_validation_fails_when_a_residual_scores_a_partial_month():
    residuals = pd.DataFrame(
        [{"stock_code": "AAA", "target_month": "2011-12", "horizon": 1, "residual": 1.0}]
    )
    result = validate_multi_horizon(
        _valid_plan(),
        residuals,
        pd.DataFrame(columns=PERIOD_PLAN_COLUMNS),
        POLICY,
        origin=ORIGIN,
        last_full_month="2011-11",
        partial_month_clean=True,
    )
    assert not result.passed
    assert "residual_never_scores_partial" in {v.rule for v in result.violations}


# --------------------------------------------------------------------------
# against the real, committed artifacts
# --------------------------------------------------------------------------
class TestPublishedArtifacts:
    """Properties that must hold on the artifacts in the repository, not just on fixtures."""

    def test_horizon_one_reproduces_the_operational_forecast(self):
        """Horizon 1 is the operational month, produced by the same estimator from the same
        features — so it must be the *same number*, not merely a close one. If these two ever
        disagree, the screen would show two different answers to one question."""
        plan = pd.read_csv(paths.MULTI_HORIZON_PLAN, dtype={"stock_code": str})
        latest = pd.read_csv(paths.LATEST_FORECAST, dtype={"stock_code": str})
        inventory = pd.read_csv(paths.INVENTORY_PLAN, dtype={"stock_code": str})

        h1 = plan.loc[plan["horizon"] == 1]
        merged = h1.merge(latest[["stock_code", "prediction"]], on="stock_code", how="inner")
        assert len(merged) == len(latest)
        assert np.allclose(merged["forecast"], merged["prediction"], rtol=0, atol=0)

        joined = h1.merge(
            inventory[["stock_code", "sigma", "target_inventory"]],
            on="stock_code",
            suffixes=("_plan", "_operational"),
        )
        assert np.allclose(
            joined["sigma_plan"], joined["sigma_operational"], rtol=0, atol=0, equal_nan=True
        )
        assert (joined["target_inventory_plan"] == joined["target_inventory_operational"]).all()

    def test_sigma_widens_with_horizon(self):
        """A recursive horizon compounds the errors before it. If a later horizon's median sigma
        were not wider, the horizon-specific back-test would not be measuring what it claims."""
        plan = pd.read_csv(paths.MULTI_HORIZON_PLAN, dtype={"stock_code": str})
        medians = plan.groupby("horizon")["sigma"].median()

        assert list(medians.index) == sorted(medians.index)
        assert medians.is_monotonic_increasing

    def test_every_forecast_month_appears_in_the_period_plan(self):
        plan = pd.read_csv(paths.MULTI_HORIZON_PLAN, dtype={"stock_code": str})
        periods = pd.read_csv(paths.PERIOD_PLAN, dtype={"stock_code": str})

        months = periods.loc[periods["period_type"] == PERIOD_MONTH, "period"]
        assert set(plan["target_month"]) <= set(months)

    def test_quarter_totals_equal_the_sum_of_their_months(self):
        """The screen reads quarter rows directly, so the summation has to be right in the file."""
        periods = pd.read_csv(paths.PERIOD_PLAN, dtype={"stock_code": str})
        months = periods.loc[periods["period_type"] == PERIOD_MONTH].copy()
        quarters = periods.loc[periods["period_type"] == PERIOD_QUARTER]

        months["quarter"] = months["period"].map(quarter_of)
        expected = months.groupby(["stock_code", "quarter"])["target_inventory"].sum()
        actual = quarters.set_index(["stock_code", "period"])["target_inventory"]

        joined = pd.concat([expected.rename("expected"), actual.rename("actual")], axis=1)
        assert joined["expected"].notna().all() and joined["actual"].notna().all()
        assert (joined["expected"] == joined["actual"]).all()
