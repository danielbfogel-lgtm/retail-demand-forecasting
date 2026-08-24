"""Recursive multi-month forecasting and the month/quarter stocking plan (US-40, PRD §32).

**The model is still one-step-ahead.** :mod:`pipeline.models` trains a single model that predicts
the month immediately after its forecast origin, and nothing here changes that. Horizons beyond the
first are produced *recursively*: the forecast for ``origin + 1`` is written back into the panel as
that month's ``units_sold``, the §17 features are rebuilt from the amended panel, and the same model
predicts ``origin + 2`` — and so on. Every horizon therefore compounds the error of the horizons
before it, which is why :func:`horizon_residuals` measures a **separate σ for every horizon** rather
than reusing the one-step-ahead σ of :mod:`pipeline.sigma`, and why ``max_horizon`` is a config
ceiling (``model_config.yaml -> multi_horizon.max_horizon``) rather than an open-ended loop.

**The partial month never enters a feature window.** December 2011 is partial (§8) and the panel
carries its truncated actuals. From the first forecast month onward this module *overwrites* the
panel's ``units_sold`` with the model's own forecast — so a recursive horizon reads the forecast for
December, never the nine days of actual sales recorded there. That is the boundary §2.5 draws for
metrics, applied to features, and :func:`validate_multi_horizon` proves it by perturbation rather
than asserting it.

**Two artifacts.**

* ``multi_horizon_plan.csv`` — one row per product × horizon: the forecast, that horizon's σ, and
  the §28 Recommended Target Inventory built from them.
* ``period_plan.csv`` — the stocking requirement per *period*, which is what a planner actually
  asks for. Month rows cover the hold-out months (from the US-21 simulation, where the actual is
  known) and the forecast months from this module. Quarter rows are the **sum of their three
  monthly target inventories** — the operational reading of a monthly re-order policy
  (``inventory_policy.yaml -> lead_time_months`` is 1: stock is set once a month), never a separate
  quarterly model, and never a quarterly σ this data cannot support. A quarter with fewer than
  three covered months is emitted with ``complete = False`` and the months it does cover, never
  silently summed as if whole.

Nothing here re-derives a number another module owns: ``safety_stock`` and ``target_inventory`` come
from :mod:`pipeline.inventory`, σ from :mod:`pipeline.sigma` through
:func:`pipeline.latest_forecast.operational_sigma`, and the features from
:func:`pipeline.features.build_features_for_origin` — the same code path the model was trained on.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline import paths
from pipeline.config import (
    CleaningConfig,
    InventoryPolicy,
    ModelConfig,
    load_cleaning_config,
    load_inventory_policy,
    load_model_config,
)
from pipeline.contract import read_panel
from pipeline.features import build_features_for_origin
from pipeline.inventory import POLICY_FORECAST_PLUS_SS, safety_stock, target_inventory
from pipeline.latest_forecast import (
    STATUS_FORECAST,
    BaselineForecaster,
    champion_id,
    operational_origin,
    operational_sigma,
    resolve_champion,
)
from pipeline.models import make_model
from pipeline.quarterly import quarter_label
from pipeline.run_context import RunContext
from pipeline.validation import ValidationResult, Violation

#: Step name on ``ctx.step(...)``, in log lines and on every :class:`Violation`.
STEP_NAME = "multi_horizon_forecast"

#: Column order of ``artifacts/forecasts/multi_horizon_plan.csv``.
MULTI_HORIZON_COLUMNS: list[str] = [
    "stock_code",
    "description",
    "forecast_origin",
    "horizon",
    "target_month",
    "model",
    "forecast",
    "sigma",
    "sigma_source",
    "n_residuals_product",
    "z",
    "safety_stock",
    "target_inventory",
    "abc_class",
    "status",
    "run_id",
]

#: Column order of ``artifacts/forecasts/period_plan.csv``.
PERIOD_PLAN_COLUMNS: list[str] = [
    "stock_code",
    "description",
    "abc_class",
    "period_type",
    "period",
    "model",
    "forecast",
    "safety_stock",
    "target_inventory",
    "actual",
    "n_months",
    "months_included",
    "complete",
    "source",
    "run_id",
]

#: ``period_type`` values — a single month the plan covers, or a calendar quarter of such months.
PERIOD_MONTH = "month"
PERIOD_QUARTER = "quarter"

#: ``source`` values: which stage produced the monthly rows behind a period.
SOURCE_HOLDOUT = "holdout_simulation"
SOURCE_FORECAST = "multi_horizon_forecast"
SOURCE_MIXED = "mixed"

#: Months in a calendar quarter — the completeness rule, matching :mod:`pipeline.quarterly`.
MONTHS_PER_QUARTER = 3

#: Panel columns neither predicted nor carried forward: never observed in a forecast month, so they
#: are zeroed rather than copied. None of them is a §17 feature.
_ZEROED_IN_FORECAST_MONTH: tuple[str, ...] = (
    "gross_revenue",
    "sale_line_count",
    "customer_count",
    "max_line_qty",
    "returned_units",
)

#: Perturbation applied to a partial month's measurements by :func:`partial_month_unreachable`.
_PERTURB_FACTOR = 7.0
_PERTURB_OFFSET = 13.0


def _repo_relative(path: Path) -> Path:
    """Repo-relative form of a canonical path constant (``docs/interfaces.md`` §6 rule 12)."""
    return path.relative_to(paths.PROJECT_ROOT)


def _shift_month(month: str, months: int) -> str:
    """``"2011-11"`` shifted by ``months`` calendar months, as ``YYYY-MM``."""
    return str(pd.Period(month, freq="M") + months)


def horizon_months(origin: str, max_horizon: int) -> list[str]:
    """The target months of horizons ``1 … max_horizon`` measured from ``origin``."""
    if max_horizon < 1:
        raise ValueError(f"max_horizon must be at least 1, got {max_horizon}")
    return [_shift_month(origin, h) for h in range(1, max_horizon + 1)]


def quarter_of(month: str) -> str:
    """The calendar quarter a month falls in, as ``YYYY-Qn``.

    Delegates to :func:`pipeline.quarterly.quarter_label` rather than formatting a period here:
    ``quarterly_forecast.csv`` already labels quarters that way, and two spellings of the same
    quarter (``2011-Q4`` and pandas' own ``2011Q4``) would make the two views impossible to join.
    """
    return quarter_label(month)


# --------------------------------------------------------------------------
# panel amendment — the one place a forecast becomes an input
# --------------------------------------------------------------------------
def ensure_month(panel_df: pd.DataFrame, month: str) -> pd.DataFrame:
    """``panel_df`` with a row at ``month`` for every product, added only where one is missing.

    ``pipeline.features`` builds its grid over ``min(month) … max(month)``, so a target month with
    no panel row produces no feature row at all. The added row's ``units_sold`` is a placeholder
    zero: no feature reads month ``t`` (every window is shifted one month back), and
    :func:`set_forecast_units` replaces it with the model's forecast before the *next* horizon
    reads it as ``lag_1``.
    """
    months = panel_df["month"].astype(str)
    if (months == month).any():
        return panel_df

    previous = _shift_month(month, -1)
    template = panel_df.loc[months == previous].copy()
    if template.empty:
        raise ValueError(
            f"cannot extend the panel to {month}: it has no rows at {previous} to extend from"
        )

    template["month"] = month
    template["units_sold"] = 0.0
    for column in _ZEROED_IN_FORECAST_MONTH:
        if column in template.columns:
            template[column] = 0
    if "is_partial_month" in template.columns:
        template["is_partial_month"] = False

    extended = pd.concat([panel_df, template], ignore_index=True)
    return extended.sort_values(["stock_code", "month"], kind="mergesort").reset_index(drop=True)


def set_forecast_units(
    panel_df: pd.DataFrame,
    month: str,
    forecasts: pd.Series,
    carry_forward_columns: list[str],
) -> pd.DataFrame:
    """``panel_df`` with ``month``'s measurements replaced by the model's forecast.

    Products without a forecast (inactive at that month, §14) get ``0.0`` — not their recorded
    actual. That wholesale replacement is what makes the partial December unreachable: no later
    horizon can read a truncated actual as its ``lag_1``. The columns in ``carry_forward_columns``
    hold their previous month's value, because ``invoice_count_lag_1`` and ``avg_unit_price_lag_1``
    are §17 features that no demand model predicts — carrying the last observed value is a stated
    assumption, not a measurement.
    """
    frame = panel_df.copy()
    # The panel's units are integer counts; a forecast is not, and assigning one into an int64
    # column is a silent-truncation trap (pandas warns today, raises tomorrow).
    frame["units_sold"] = frame["units_sold"].astype(float)
    months = frame["month"].astype(str)
    mask = (months == month).to_numpy()
    if not mask.any():
        raise ValueError(f"panel has no rows at {month}; call ensure_month first")

    codes = frame.loc[mask, "stock_code"].astype(str)
    frame.loc[mask, "units_sold"] = codes.map(forecasts).fillna(0.0).astype(float).to_numpy()

    previous_mask = (months == _shift_month(month, -1)).to_numpy()
    if previous_mask.any():
        previous_rows = frame.loc[previous_mask].copy()
        previous_rows.index = previous_rows["stock_code"].astype(str)
        for column in carry_forward_columns:
            if column in frame.columns:
                carried = codes.map(previous_rows[column])
                frame.loc[mask, column] = carried.fillna(0).to_numpy()

    for column in _ZEROED_IN_FORECAST_MONTH:
        if column in frame.columns:
            frame.loc[mask, column] = 0
    if "is_partial_month" in frame.columns:
        frame.loc[mask, "is_partial_month"] = False
    return frame


# --------------------------------------------------------------------------
# the recursion itself — pure: no ctx, no disk
# --------------------------------------------------------------------------
def fit_champion_at(
    features_df: pd.DataFrame, champion: str, origin: str, cfg: ModelConfig, seed: int
) -> Any:
    """The champion fitted on every target known at ``origin`` — the §16 leakage boundary.

    Mirrors :func:`pipeline.models.fit_predict_one_origin`'s training filter exactly
    (``target_month <= origin``) but keeps the fitted estimator, because a recursive horizon has to
    call ``predict`` again on features that do not exist yet at fit time. A baseline champion is a
    rule, not a fit, and comes back as :class:`pipeline.latest_forecast.BaselineForecaster`.
    """
    if cfg.models[champion].kind == "baseline":
        return BaselineForecaster(champion)

    months = features_df["target_month"].astype(str)
    train_rows = features_df.loc[months <= origin]
    if train_rows.empty:
        raise ValueError(f"no training rows at or before origin {origin}")
    model = make_model(champion, cfg, seed)
    model.fit(train_rows[list(cfg.features)], train_rows["y"].astype(float))
    return model


def recursive_forecast(
    panel_df: pd.DataFrame,
    model: Any,
    cfg: ModelConfig,
    origin: str,
    max_horizon: int,
    *,
    champion: str,
) -> pd.DataFrame:
    """Forecast ``origin + 1 … origin + max_horizon`` by feeding each forecast back into the panel.

    Returns ``stock_code, forecast_origin, horizon, step_origin, target_month, model, forecast`` —
    one row per product active at each horizon, the forecast clipped at zero for business use
    exactly as :func:`pipeline.models.fit_predict_one_origin` clips the back-test's.

    ``step_origin`` is the origin the *features* of that row were built at (``origin + h - 1``);
    ``forecast_origin`` stays the real one, the last month backed by data. Recording both is what
    makes a horizon-2 row legible: its features are built from a month that is itself a forecast.
    """
    work = panel_df.copy()
    work["units_sold"] = work["units_sold"].astype(float)
    carry = list(cfg.multi_horizon.carry_forward_columns)
    frames: list[pd.DataFrame] = []

    for horizon in range(1, max_horizon + 1):
        step_origin = _shift_month(origin, horizon - 1)
        target = _shift_month(origin, horizon)

        work = ensure_month(work, target)
        features = build_features_for_origin(work, step_origin, cfg.active_rule.k, cfg)
        if features.empty:
            break

        raw = np.asarray(model.predict(features[list(cfg.features)]), dtype=float)
        forecast = np.clip(raw, 0, None)
        codes = features["stock_code"].astype(str)

        frames.append(
            pd.DataFrame(
                {
                    "stock_code": codes.to_numpy(),
                    "forecast_origin": origin,
                    "horizon": horizon,
                    "step_origin": step_origin,
                    "target_month": target,
                    "model": champion,
                    "forecast": forecast,
                }
            )
        )

        if horizon < max_horizon:
            work = set_forecast_units(
                work, target, pd.Series(forecast, index=codes.to_numpy()), carry
            )

    if not frames:
        return pd.DataFrame(
            columns=[
                "stock_code",
                "forecast_origin",
                "horizon",
                "step_origin",
                "target_month",
                "model",
                "forecast",
            ]
        )
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["horizon", "stock_code"], kind="mergesort")
        .reset_index(drop=True)
    )


def partial_month_unreachable(
    panel_df: pd.DataFrame,
    model: Any,
    cfg: ModelConfig,
    origin: str,
    max_horizon: int,
    cleaning_cfg: CleaningConfig,
    *,
    champion: str,
) -> bool:
    """Do the recursive forecasts survive corrupting every partial month's measurements?

    The proof that no horizon reads December 2011's truncated actuals: perturb them, re-run the
    whole recursion, and require an identical frame. Mirrors
    :func:`pipeline.latest_forecast.validate_operational_inputs`'s ``partial_month_not_used`` rule,
    extended to every horizon rather than only the first.
    """
    partial_months = [str(month) for month in cleaning_cfg.raw.partial_months]
    if not partial_months:
        return True

    corrupted = panel_df.copy()
    mask = corrupted["month"].astype(str).isin(partial_months).to_numpy()
    for column in ("units_sold", *cfg.multi_horizon.carry_forward_columns):
        if column in corrupted.columns:
            corrupted[column] = corrupted[column].astype(float)
            corrupted.loc[mask, column] = (
                corrupted.loc[mask, column] * _PERTURB_FACTOR + _PERTURB_OFFSET
            )

    clean = recursive_forecast(panel_df, model, cfg, origin, max_horizon, champion=champion)
    perturbed = recursive_forecast(corrupted, model, cfg, origin, max_horizon, champion=champion)
    if clean.shape != perturbed.shape:
        return False
    return bool(
        np.allclose(
            clean["forecast"].to_numpy(dtype=float),
            perturbed["forecast"].to_numpy(dtype=float),
            rtol=0,
            atol=0,
        )
    )


# --------------------------------------------------------------------------
# horizon-specific σ — a recursive back-test, one residual set per horizon
# --------------------------------------------------------------------------
def backtest_origins(cfg: ModelConfig) -> list[str]:
    """Every rolling origin of §22, ``backtest.first_origin … backtest.last_origin`` inclusive."""
    first = pd.Period(cfg.backtest.first_origin, freq="M")
    last = pd.Period(cfg.backtest.last_origin, freq="M")
    return [str(period) for period in pd.period_range(first, last, freq="M")]


def horizon_residuals(
    panel_df: pd.DataFrame,
    features_df: pd.DataFrame,
    cfg: ModelConfig,
    champion: str,
    last_full_month: str,
    max_horizon: int,
) -> pd.DataFrame:
    """Out-of-sample residuals for every horizon, from a recursive back-test (§26).

    At each rolling origin the champion is refitted on the data known there, then run forward
    recursively; a horizon-``h`` residual is ``actual(origin + h) − forecast_h``. Only targets at or
    before ``last_full_month`` produce a residual, so the partial December is never scored (§2.5) —
    the rule the one-step-ahead back-test already obeys.

    Returns ``stock_code, forecast_origin, horizon, target_month, model, actual, forecast,
    residual``. Feeding these into :func:`horizon_sigma` per horizon is what gives horizon 3 its
    own, wider σ instead of borrowing horizon 1's.
    """
    actuals = panel_df[["stock_code", "month", "units_sold"]].copy()
    actuals["stock_code"] = actuals["stock_code"].astype(str)
    actuals["month"] = actuals["month"].astype(str)
    actuals = actuals.rename(columns={"month": "target_month", "units_sold": "actual"})
    actuals["actual"] = actuals["actual"].astype(float)

    columns = [
        "stock_code",
        "forecast_origin",
        "horizon",
        "target_month",
        "model",
        "actual",
        "forecast",
        "residual",
    ]

    # The configured back-test window is the *intent*; the data is the fact. Two limits narrow it,
    # and both are read rather than re-derived: the panel's own month range, and the first target
    # the feature builder can produce (``lag_3`` needs three months of history, so the earliest
    # origins of a short panel — the CI sample fixture, a trimmed extract — have no feature row at
    # all and would raise inside build_features_for_origin). Fewer usable origins is a smaller
    # residual set, not an error.
    panel_months = panel_df["month"].astype(str)
    first_panel_month, last_panel_month = panel_months.min(), panel_months.max()
    first_feature_target = (
        str(features_df["target_month"].astype(str).min()) if not features_df.empty else None
    )

    frames: list[pd.DataFrame] = []
    for origin in backtest_origins(cfg):
        if not first_panel_month <= origin <= last_panel_month:
            continue
        if first_feature_target is not None and _shift_month(origin, 1) < first_feature_target:
            continue
        reachable = sum(
            1 for h in range(1, max_horizon + 1) if _shift_month(origin, h) <= last_full_month
        )
        if reachable == 0:
            continue

        known_panel = panel_df.loc[panel_df["month"].astype(str) <= origin].copy()
        model = fit_champion_at(features_df, champion, origin, cfg, cfg.seed)
        predictions = recursive_forecast(
            known_panel, model, cfg, origin, reachable, champion=champion
        )
        if predictions.empty:
            continue
        frames.append(predictions.drop(columns=["step_origin"]))

    if not frames:
        return pd.DataFrame(columns=columns)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.merge(actuals, on=["stock_code", "target_month"], how="inner")
    combined = combined.loc[combined["target_month"].astype(str) <= last_full_month]
    combined["residual"] = combined["actual"].astype(float) - combined["forecast"].astype(float)
    return (
        combined.sort_values(["horizon", "target_month", "stock_code"], kind="mergesort")
        .reset_index(drop=True)[columns]
    )


def horizon_sigma(
    residuals_df: pd.DataFrame,
    abc_train_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    champion: str,
    policy_cfg: InventoryPolicy,
) -> pd.DataFrame:
    """σ per product for one horizon, through the §27 product → ABC → global fallback.

    Delegates to :func:`pipeline.latest_forecast.operational_sigma`, which is already the "price a
    month the back-test does not reach" case: the horizon's own residuals are its history and the
    horizon's target month is the evaluation month. Nothing about the σ *definition* changes — only
    which residuals are fed in, which is the whole point of a horizon-specific σ.
    """
    empty = pd.DataFrame(
        columns=["stock_code", "sigma", "sigma_source", "n_residuals_product"]
    )
    if forecast_df.empty:
        return empty

    history = residuals_df.loc[:, ["stock_code", "target_month", "model", "residual"]].copy()
    table = operational_sigma(history, abc_train_df, forecast_df, champion, policy_cfg)
    if table.empty:
        return empty
    return table[["stock_code", "sigma", "sigma_source", "n_residuals_product"]]


# --------------------------------------------------------------------------
# the plan — forecast + σ -> Recommended Target Inventory, per horizon
# --------------------------------------------------------------------------
def _descriptions(panel_df: pd.DataFrame) -> pd.Series:
    """Last non-null description per product — display only, exactly as US-23 uses it."""
    frame = panel_df[["stock_code", "month", "description"]].copy()
    frame["stock_code"] = frame["stock_code"].astype(str)
    frame = frame.sort_values(["stock_code", "month"], kind="mergesort")
    return frame.groupby("stock_code")["description"].last()


def build_multi_horizon_plan(
    forecast_df: pd.DataFrame,
    residuals_df: pd.DataFrame,
    abc_train_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    policy_cfg: InventoryPolicy,
    *,
    champion: str,
    run_id: str,
) -> pd.DataFrame:
    """One row per product × horizon: forecast, that horizon's σ, safety stock, target inventory.

    ``safety_stock`` and ``target_inventory`` are :mod:`pipeline.inventory`'s functions, never
    restated here — one definition of §25 and §28 for the hold-out simulation, the operational plan
    and every recursive horizon alike.
    """
    if forecast_df.empty:
        return pd.DataFrame(columns=MULTI_HORIZON_COLUMNS)

    abc_lookup = abc_train_df[["stock_code", "abc_class"]].copy()
    abc_lookup["stock_code"] = abc_lookup["stock_code"].astype(str)
    descriptions = _descriptions(panel_df)

    frames: list[pd.DataFrame] = []
    for horizon in sorted(forecast_df["horizon"].unique()):
        rows = forecast_df.loc[forecast_df["horizon"] == horizon].copy()
        rows["stock_code"] = rows["stock_code"].astype(str)
        sigma_rows = horizon_sigma(
            residuals_df.loc[residuals_df["horizon"] == horizon],
            abc_train_df,
            rows,
            champion,
            policy_cfg,
        )
        merged = rows.merge(sigma_rows, on="stock_code", how="left")
        frames.append(merged)

    plan = pd.concat(frames, ignore_index=True)
    plan["z"] = policy_cfg.z
    plan["safety_stock"] = safety_stock(plan["sigma"].astype(float), policy_cfg.z)
    plan["target_inventory"] = target_inventory(
        plan["forecast"].astype(float), plan["sigma"].astype(float), policy_cfg.z
    )
    plan["description"] = plan["stock_code"].map(descriptions)
    plan = plan.merge(abc_lookup, on="stock_code", how="left")
    plan["status"] = STATUS_FORECAST
    plan["run_id"] = run_id
    plan["n_residuals_product"] = plan["n_residuals_product"].astype("Int64")
    plan["target_inventory"] = plan["target_inventory"].astype("Int64")

    return (
        plan.sort_values(["horizon", "stock_code"], kind="mergesort")
        .reset_index(drop=True)[MULTI_HORIZON_COLUMNS]
    )


def holdout_month_rows(
    sim_rows_df: pd.DataFrame, policy_cfg: InventoryPolicy, *, champion: str
) -> pd.DataFrame:
    """The champion's hold-out months as monthly stocking rows (US-21's simulation, §29).

    Only the ``forecast_plus_ss`` policy at the configured ``z`` is a stocking recommendation; the
    ``forecast_only`` rows exist to quantify what a no-safety-stock policy would have cost and are
    not a plan. ``actual`` comes along because these months already happened — that is what lets the
    screen show a past period's requirement next to what really sold.
    """
    columns = [
        "stock_code",
        "abc_class",
        "period",
        "model",
        "forecast",
        "safety_stock",
        "target_inventory",
        "actual",
        "source",
    ]
    if sim_rows_df is None or sim_rows_df.empty:
        return pd.DataFrame(columns=columns)

    rows = sim_rows_df.loc[
        (sim_rows_df["model"] == champion)
        & (sim_rows_df["policy"] == POLICY_FORECAST_PLUS_SS)
        & (np.isclose(sim_rows_df["z"].astype(float), policy_cfg.z))
    ].copy()
    if rows.empty:
        return pd.DataFrame(columns=columns)

    rows["stock_code"] = rows["stock_code"].astype(str)
    rows["period"] = rows["target_month"].astype(str)
    rows["safety_stock"] = safety_stock(rows["sigma"].astype(float), policy_cfg.z)
    rows["source"] = SOURCE_HOLDOUT
    return rows[columns].reset_index(drop=True)


def forecast_month_rows(plan_df: pd.DataFrame) -> pd.DataFrame:
    """The recursive horizons as monthly stocking rows — no ``actual``: they have not happened."""
    columns = [
        "stock_code",
        "abc_class",
        "period",
        "model",
        "forecast",
        "safety_stock",
        "target_inventory",
        "actual",
        "source",
    ]
    if plan_df.empty:
        return pd.DataFrame(columns=columns)

    rows = plan_df.copy()
    rows["stock_code"] = rows["stock_code"].astype(str)
    rows["period"] = rows["target_month"].astype(str)
    rows["actual"] = np.nan
    rows["source"] = SOURCE_FORECAST
    return rows[columns].reset_index(drop=True)


def build_period_plan(
    month_rows: pd.DataFrame, panel_df: pd.DataFrame, *, run_id: str
) -> pd.DataFrame:
    """Month rows plus their quarterly sums — the frame the Product Forecasts screen reads.

    A quarter's requirement is the **sum of its monthly target inventories**: with
    ``lead_time_months = 1`` the stock level is re-set every month, so the units a product needs
    across a quarter is what its three monthly plans ask for. The safety stock is therefore counted
    once per month, deliberately — it is the buffer the policy actually holds, not a single
    three-month buffer, which would need a quarterly σ this back-test cannot estimate per product.
    ``complete`` says whether all three calendar months are covered; ``months_included`` names them,
    so a partial quarter can never be read as a whole one.
    """
    if month_rows.empty:
        return pd.DataFrame(columns=PERIOD_PLAN_COLUMNS)

    descriptions = _descriptions(panel_df)

    months = month_rows.copy()
    months["period_type"] = PERIOD_MONTH
    months["n_months"] = 1
    months["months_included"] = months["period"]
    months["complete"] = True

    quarter_key = months["period"].map(quarter_of)
    grouped = months.assign(quarter=quarter_key).groupby(
        ["stock_code", "abc_class", "model", "quarter"], sort=False, dropna=False
    )
    quarters = grouped.agg(
        forecast=("forecast", "sum"),
        safety_stock=("safety_stock", "sum"),
        target_inventory=("target_inventory", "sum"),
        actual=("actual", "sum"),
        n_months=("period", "size"),
        months_included=("period", lambda values: ";".join(sorted(values))),
        n_actual=("actual", "count"),
        sources=("source", lambda values: sorted(set(values))),
    ).reset_index()

    quarters["period_type"] = PERIOD_QUARTER
    quarters = quarters.rename(columns={"quarter": "period"})
    quarters["complete"] = quarters["n_months"] == MONTHS_PER_QUARTER
    # A quarter's actual is only a number when every month of it is known; a partly-forecast
    # quarter reports no actual rather than a sum of the months that happen to have one.
    quarters["actual"] = quarters["actual"].where(
        quarters["n_actual"] == quarters["n_months"], np.nan
    )
    quarters["source"] = quarters["sources"].map(
        lambda values: values[0] if len(values) == 1 else SOURCE_MIXED
    )
    quarters = quarters.drop(columns=["n_actual", "sources"])

    combined = pd.concat([months, quarters], ignore_index=True)
    combined["description"] = combined["stock_code"].map(descriptions)
    combined["run_id"] = run_id
    combined["target_inventory"] = combined["target_inventory"].astype("Int64")
    combined["n_months"] = combined["n_months"].astype("Int64")

    return (
        combined.sort_values(
            ["period_type", "period", "stock_code"], kind="mergesort"
        )
        .reset_index(drop=True)[PERIOD_PLAN_COLUMNS]
    )


# --------------------------------------------------------------------------
# validation — pure: returns a ValidationResult, writes nothing (§6 rule 5)
# --------------------------------------------------------------------------
def validate_multi_horizon(
    plan_df: pd.DataFrame,
    residuals_df: pd.DataFrame,
    period_df: pd.DataFrame,
    policy_cfg: InventoryPolicy,
    *,
    origin: str,
    last_full_month: str,
    partial_month_clean: bool,
) -> ValidationResult:
    """Five invariants of the recursive plan (pure — no ``ctx``, no disk).

    * ``horizon_target_month`` — every row's target is exactly ``origin + horizon``.
    * ``partial_month_not_used`` — the perturbation result of :func:`partial_month_unreachable`.
    * ``residual_never_scores_partial`` — no residual is measured against a partial month.
    * ``target_inventory_formula`` — the stored number is ``ceil(max(0, forecast + z·σ))``.
    * ``quarter_completeness`` — a quarter row's ``complete`` matches the months it names.
    """
    violations: list[Violation] = []

    if not plan_df.empty:
        expected = plan_df.apply(
            lambda row: _shift_month(str(row["forecast_origin"]), int(row["horizon"])), axis=1
        )
        mismatched = plan_df.loc[expected != plan_df["target_month"].astype(str)]
        if not mismatched.empty:
            violations.append(
                Violation(
                    step=STEP_NAME,
                    rule="horizon_target_month",
                    message="target_month must be forecast_origin + horizon",
                    count=int(len(mismatched)),
                    examples=mismatched["stock_code"].astype(str).head(5).tolist(),
                )
            )

        recomputed = target_inventory(
            plan_df["forecast"].astype(float), plan_df["sigma"].astype(float), policy_cfg.z
        )
        stored = plan_df["target_inventory"].astype("Int64")
        differs = stored.to_numpy(dtype="float64") != np.asarray(recomputed, dtype="float64")
        if bool(differs.any()):
            violations.append(
                Violation(
                    step=STEP_NAME,
                    rule="target_inventory_formula",
                    message="target_inventory must equal ceil(max(0, forecast + z * sigma))",
                    count=int(differs.sum()),
                )
            )

    if not partial_month_clean:
        violations.append(
            Violation(
                step=STEP_NAME,
                rule="partial_month_not_used",
                message=(
                    "perturbing the partial month's measurements changed a recursive forecast: "
                    "a horizon is reading the partial month instead of its own forecast"
                ),
            )
        )

    if not residuals_df.empty:
        scored_partial = residuals_df.loc[
            residuals_df["target_month"].astype(str) > last_full_month
        ]
        if not scored_partial.empty:
            violations.append(
                Violation(
                    step=STEP_NAME,
                    rule="residual_never_scores_partial",
                    message=f"a residual was measured after the last full month {last_full_month}",
                    count=int(len(scored_partial)),
                )
            )

    if not period_df.empty:
        quarters = period_df.loc[period_df["period_type"] == PERIOD_QUARTER]
        named = quarters["months_included"].astype(str).str.split(";").map(len)
        inconsistent = quarters.loc[
            quarters["complete"].astype(bool) != (named == MONTHS_PER_QUARTER)
        ]
        if not inconsistent.empty:
            violations.append(
                Violation(
                    step=STEP_NAME,
                    rule="quarter_completeness",
                    message="complete must be true exactly when three months are named",
                    count=int(len(inconsistent)),
                )
            )

    return ValidationResult(
        step=STEP_NAME,
        passed=not violations,
        violations=violations,
        checked_rows=int(len(plan_df)),
        extra={
            "origin": origin,
            "horizons": sorted(int(h) for h in plan_df["horizon"].unique())
            if not plan_df.empty
            else [],
            "last_full_month": last_full_month,
        },
    )


# --------------------------------------------------------------------------
# writers — every artifact through ctx.out() (§6 rule 1)
# --------------------------------------------------------------------------
def write_multi_horizon_plan(frame: pd.DataFrame, ctx: RunContext) -> Path:
    """Write ``multi_horizon_plan.csv`` and register it on the run."""
    destination = ctx.out(_repo_relative(paths.MULTI_HORIZON_PLAN))
    frame.to_csv(
        destination, index=False, float_format="%.6f", lineterminator="\n", encoding="utf-8"
    )
    ctx.record_artifact("multi_horizon_plan", _repo_relative(paths.MULTI_HORIZON_PLAN))
    return destination


def write_period_plan(frame: pd.DataFrame, ctx: RunContext) -> Path:
    """Write ``period_plan.csv`` and register it on the run."""
    destination = ctx.out(_repo_relative(paths.PERIOD_PLAN))
    frame.to_csv(
        destination, index=False, float_format="%.6f", lineterminator="\n", encoding="utf-8"
    )
    ctx.record_artifact("period_plan", _repo_relative(paths.PERIOD_PLAN))
    return destination


# --------------------------------------------------------------------------
# the Flow entry point
# --------------------------------------------------------------------------
def run_multi_horizon(
    cfg: ModelConfig,
    ctx: RunContext,
    *,
    panel_df: pd.DataFrame,
    features_df: pd.DataFrame,
    abc_train_df: pd.DataFrame,
    sim_rows_df: pd.DataFrame,
    champion: str,
    champion_model: Any,
    policy_cfg: InventoryPolicy | None = None,
    cleaning_cfg: CleaningConfig | None = None,
) -> dict[str, Any]:
    """Recursive horizons, their σ, the plan and the period view — the Flow's step-8 extension.

    Every input is a DataFrame the caller already holds: under the Flow the upstream files are still
    in ``artifacts/_staging/<run_id>/`` and reading ``paths.*`` here would silently pick up the
    previous run's (``docs/interfaces.md`` §6 rule 7). ``champion_model`` is the estimator US-23
    already refitted through the origin — refitting it here would double the cost and risk drift.

    Opens its own ``ctx.step(...)``: it calls ``ctx.log_rows``.
    """
    resolved_policy = load_inventory_policy() if policy_cfg is None else policy_cfg
    resolved_cleaning = load_cleaning_config() if cleaning_cfg is None else cleaning_cfg
    origin = operational_origin(resolved_cleaning)
    last_full_month = resolved_cleaning.raw.last_full_month
    max_horizon = cfg.multi_horizon.max_horizon

    with ctx.step(STEP_NAME):
        forecasts = recursive_forecast(
            panel_df, champion_model, cfg, origin, max_horizon, champion=champion
        )
        residuals = horizon_residuals(
            panel_df, features_df, cfg, champion, last_full_month, max_horizon
        )
        plan = build_multi_horizon_plan(
            forecasts,
            residuals,
            abc_train_df,
            panel_df,
            resolved_policy,
            champion=champion,
            run_id=ctx.run_id,
        )
        month_rows = pd.concat(
            [
                holdout_month_rows(sim_rows_df, resolved_policy, champion=champion),
                forecast_month_rows(plan),
            ],
            ignore_index=True,
        )
        period_plan = build_period_plan(month_rows, panel_df, run_id=ctx.run_id)

        partial_clean = partial_month_unreachable(
            panel_df,
            champion_model,
            cfg,
            origin,
            max_horizon,
            resolved_cleaning,
            champion=champion,
        )
        validation = validate_multi_horizon(
            plan,
            residuals,
            period_plan,
            resolved_policy,
            origin=origin,
            last_full_month=last_full_month,
            partial_month_clean=partial_clean,
        )

        ctx.log_rows(
            "multi_horizon_rows",
            before=int(len(forecasts)),
            removed=int(len(forecasts) - len(plan)),
            after=int(len(plan)),
        )
        for horizon in sorted(plan["horizon"].unique()) if not plan.empty else []:
            rows = plan.loc[plan["horizon"] == horizon]
            ctx.log_rows(
                f"multi_horizon_h{int(horizon)}",
                before=int(len(rows)),
                removed=0,
                after=int(len(rows)),
            )

        write_multi_horizon_plan(plan, ctx)
        write_period_plan(period_plan, ctx)

    return {
        "forecasts": forecasts,
        "residuals": residuals,
        "multi_horizon_plan": plan,
        "period_plan": period_plan,
        "validation": validation,
    }


# --------------------------------------------------------------------------
# CLI — standalone, reads the canonical paths (correct only when staging=False)
# --------------------------------------------------------------------------
def _read_csv(path: Path, what: str) -> pd.DataFrame:
    """One published artifact, ``stock_code`` kept a string (§8)."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found ({what}). Run `python -m pipeline --no-llm` first."
        )
    return pd.read_csv(path, dtype={"stock_code": str})


def run(argv: list[str] | None = None) -> int:
    """``python -m pipeline.multi_horizon`` — rebuild the plan from the published artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    ctx = RunContext.start(mode="no-llm")
    try:
        cfg = load_model_config()
        panel_df = read_panel(paths.CLEAN_DATA)
        features_df = _read_csv(paths.FEATURES, "features.csv")
        abc_train_df = _read_csv(paths.EVAL_TABLES_DIR / "abc_train.csv", "abc_train.csv")
        sim_rows_df = _read_csv(paths.HOLDOUT_SIMULATION_ROWS, "holdout_simulation_rows.csv")
        champion = champion_id(resolve_champion(ctx))

        cleaning_cfg = load_cleaning_config()
        origin = operational_origin(cleaning_cfg)
        champion_model = fit_champion_at(features_df, champion, origin, cfg, cfg.seed)

        result = run_multi_horizon(
            cfg,
            ctx,
            panel_df=panel_df,
            features_df=features_df,
            abc_train_df=abc_train_df,
            sim_rows_df=sim_rows_df,
            champion=champion,
            champion_model=champion_model,
        )
    except Exception:  # pragma: no cover - CLI guard, mirrors pipeline.quarterly.run
        ctx.finish(status="failed")
        raise

    validation: ValidationResult = result["validation"]
    if not validation.passed:
        ctx.finish(status="failed")
        return 2
    ctx.finish()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
