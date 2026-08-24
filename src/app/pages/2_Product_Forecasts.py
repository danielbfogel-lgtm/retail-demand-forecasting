"""Screen 2 — Product Forecasts (US-28, US-40, PRD §33.2). Table, filters and CSV download.

Three views over the same product universe:

* **Next month** — the operational plan (``inventory_plan.csv``): the single one-step-ahead forecast
  and its Recommended Target Inventory, for every product in the panel including the ones that get
  no forecast and say why.
* **By month** / **By quarter** — the stocking requirement for a chosen period
  (``period_plan.csv``): the hold-out months, whose actuals are known, and the recursive forecast
  months beyond the operational one. Quarter rows are the sum of their monthly target inventories,
  summed by :mod:`pipeline.multi_horizon`, not here — this screen filters and displays.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import data_access
from app.components.status import run_status_banner
from app.components.theme import apply_theme
from pipeline.latest_forecast import STATUS_FORECAST
from pipeline.multi_horizon import (
    PERIOD_MONTH,
    PERIOD_QUARTER,
    SOURCE_FORECAST,
    SOURCE_HOLDOUT,
    SOURCE_MIXED,
)

#: "High uncertainty" default threshold (§4 of the issue). Kept as a plain page-level default
#: rather than a new ``inventory_policy.yaml -> app.*`` key: ``InventoryPolicy`` forbids unknown
#: top-level keys, so an unreviewed schema addition here would break the whole pipeline for every
#: config consumer, not just this screen (AI-33 §8 interface correction — this is the sanctioned
#: alternative).
DEFAULT_HIGH_UNCERTAINTY_RATIO = 1.0

#: The three views. The operational plan stays first and default, so the screen a reviewer already
#: knows is the one that opens.
VIEW_NEXT_MONTH = "Next month — full plan"
VIEW_BY_MONTH = "By month"
VIEW_BY_QUARTER = "By quarter"

#: Plain-language label for each ``period_plan.csv`` source.
SOURCE_LABELS = {
    SOURCE_HOLDOUT: "actuals known (hold-out month)",
    SOURCE_FORECAST: "forecast",
    SOURCE_MIXED: "part actual, part forecast",
}

st.set_page_config(page_title="Product Forecasts", layout="wide")
apply_theme()
st.title("Product Forecasts")

state = run_status_banner()
if state.status != "success":
    st.stop()


def _search_and_abc(table: pd.DataFrame, *, key_prefix: str) -> pd.Series:
    """The two filters both views share, as a boolean mask over ``table``."""
    filter_cols = st.columns([2, 1, 1])
    with filter_cols[0]:
        search = st.text_input(
            "Search by StockCode or description",
            value="",
            help="Case-insensitive substring match.",
            key=f"{key_prefix}_search",
        )
    with filter_cols[1]:
        abc_options = sorted(table["abc_class"].dropna().unique())
        selected_abc = st.multiselect(
            "ABC class", options=abc_options, default=abc_options, key=f"{key_prefix}_abc"
        )
    with filter_cols[2]:
        forecast_positive = st.toggle(
            "Forecast > 0", value=False, key=f"{key_prefix}_positive"
        )

    mask = pd.Series(True, index=table.index)
    if search:
        needle = search.strip().lower()
        mask &= table["stock_code"].str.lower().str.contains(needle, na=False) | table[
            "description"
        ].fillna("").str.lower().str.contains(needle, na=False)
    mask &= table["abc_class"].isin(selected_abc)
    if forecast_positive:
        mask &= table["forecast"] > 0
    return mask


# --------------------------------------------------------------------------
# view 1 — the operational plan (US-28, unchanged)
# --------------------------------------------------------------------------
def render_operational_plan() -> None:
    """One row per product in the panel for the operational month, forecast or not."""
    plan = data_access.load_inventory_plan()
    latest = data_access.load_latest_forecast()

    table = plan.merge(
        latest[["stock_code", "lag_1", "rolling_mean_3"]], on="stock_code", how="left"
    )

    st.subheader("Filters")
    filter_cols = st.columns([2, 1, 1, 1])

    with filter_cols[0]:
        search = st.text_input(
            "Search by StockCode or description",
            value="",
            help="Case-insensitive substring match.",
        )

    with filter_cols[1]:
        abc_options = sorted(table["abc_class"].dropna().unique())
        selected_abc = st.multiselect("ABC class", options=abc_options, default=abc_options)

    with filter_cols[2]:
        forecast_positive = st.toggle("Forecast > 0", value=False)

    with filter_cols[3]:
        high_uncertainty_only = st.toggle(
            "High uncertainty only",
            value=False,
            help="sigma_source != product OR uncertainty_ratio >= threshold",
        )

    if high_uncertainty_only:
        uncertainty_threshold = st.slider(
            "High-uncertainty ratio threshold",
            min_value=0.1,
            max_value=3.0,
            value=DEFAULT_HIGH_UNCERTAINTY_RATIO,
            step=0.1,
            help="A product counts as high uncertainty when its sigma was not measured from the "
            "product itself, or when sigma / forecast is at least this ratio.",
        )
    else:
        uncertainty_threshold = DEFAULT_HIGH_UNCERTAINTY_RATIO

    status_options = sorted(table["status"].dropna().unique())
    selected_status = st.multiselect(
        "Status",
        options=status_options,
        default=status_options,
        help='Product status: "Forecast" (an operational forecast is available), an inactive '
        "status (no sales in the configured lookback window), or "
        '"Insufficient History / New Product" (no sales observed at or before the forecast '
        "origin).",
    )

    mask = pd.Series(True, index=table.index)
    if search:
        needle = search.strip().lower()
        mask &= (
            table["stock_code"].str.lower().str.contains(needle, na=False)
            | table["description"].str.lower().str.contains(needle, na=False)
        )
    mask &= table["abc_class"].isin(selected_abc)
    mask &= table["status"].isin(selected_status)
    if forecast_positive:
        mask &= table["forecast"] > 0
    if high_uncertainty_only:
        mask &= (table["status"] == STATUS_FORECAST) & (
            (table["sigma_source"] != "product")
            | (table["uncertainty_ratio"] >= uncertainty_threshold)
        )

    filtered = table.loc[mask].reset_index(drop=True)

    st.caption(f"{len(filtered):,} of {len(table):,} products match the current filters.")

    if filtered.empty:
        st.info("No products match the current filters.")
        return

    display = pd.DataFrame(
        {
            "Product": filtered["stock_code"],
            "Description": filtered["description"],
            "Last Month": filtered["lag_1"],
            "3M Avg": filtered["rolling_mean_3"],
            "Forecast": filtered["forecast"],
            "Safety Stock": filtered["safety_stock"],
            "Recommended Target Inventory": filtered["target_inventory"],
            "Sigma source": filtered["sigma_source"],
            "ABC": filtered["abc_class"],
            "Status": filtered["status"],
        }
    )
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Last Month": st.column_config.NumberColumn(format="%d"),
            "3M Avg": st.column_config.NumberColumn(format="%.1f"),
            "Forecast": st.column_config.NumberColumn(format="%.1f"),
            "Safety Stock": st.column_config.NumberColumn(format="%.1f"),
            "Recommended Target Inventory": st.column_config.NumberColumn(format="%d"),
        },
    )

    run_log = data_access.load_run_log()
    run_id = run_log["run_id"]
    export_columns = [
        "stock_code",
        "description",
        "lag_1",
        "rolling_mean_3",
        "forecast",
        "safety_stock",
        "target_inventory",
        "sigma_source",
        "abc_class",
        "status",
        "forecast_origin",
        "target_month",
        "sigma",
        "z",
        "run_id",
    ]
    export_df = filtered.reindex(columns=export_columns)
    st.download_button(
        "Download CSV",
        data=export_df.to_csv(index=False),
        file_name=f"product_forecasts_{run_id}.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Open a product")
    codes = filtered["stock_code"].tolist()
    descriptions = dict(zip(filtered["stock_code"], filtered["description"], strict=False))
    selected_code = st.selectbox(
        "Select a product to view its detail page",
        options=codes,
        format_func=lambda code: f"{code} — {descriptions.get(code, '')}",
    )
    if st.button("Open Product Detail →"):
        st.session_state["selected_stock_code"] = selected_code
        st.switch_page("pages/3_Product_Detail.py")


# --------------------------------------------------------------------------
# views 2 and 3 — a chosen month or quarter (US-40)
# --------------------------------------------------------------------------
def render_period_view(period_type: str) -> None:
    """The stocking requirement for one month or one quarter, read from ``period_plan.csv``."""
    try:
        plan = data_access.load_period_plan()
    except FileNotFoundError:
        st.warning(
            "period_plan.csv is missing — this run predates the multi-horizon step. "
            "Re-run the pipeline to populate the month and quarter views."
        )
        return

    periods_frame = plan.loc[plan["period_type"] == period_type]
    if periods_frame.empty:
        st.info("This run produced no periods of that kind.")
        return

    periods = sorted(periods_frame["period"].unique())
    forecast_periods = sorted(
        periods_frame.loc[periods_frame["source"] != SOURCE_HOLDOUT, "period"].unique()
    )
    # Open on the first period that is not already history: a planner's question is about the
    # months ahead, and the hold-out months are there for comparison, not for ordering.
    default_period = forecast_periods[0] if forecast_periods else periods[-1]

    label = "Month" if period_type == PERIOD_MONTH else "Quarter"
    selected_period = st.selectbox(
        label,
        options=periods,
        index=periods.index(default_period),
        help="Hold-out periods carry the actual demand that followed; later periods are forecasts.",
    )

    table = periods_frame.loc[periods_frame["period"] == selected_period].reset_index(drop=True)

    sources = sorted(table["source"].unique())
    source_text = ", ".join(SOURCE_LABELS.get(source, source) for source in sources)
    covered = sorted({month for row in table["months_included"] for month in str(row).split(";")})
    st.caption(f"{selected_period} — {source_text}. Months covered: {', '.join(covered)}.")

    if period_type == PERIOD_QUARTER:
        complete = int(table["complete"].astype(bool).sum())
        if complete < len(table):
            st.info(
                f"{complete:,} of {len(table):,} products have all three months of "
                f"{selected_period} covered. The rest are partial quarters — their total covers "
                "only the months named in the table, never the whole quarter."
            )

    mask = _search_and_abc(table, key_prefix=f"period_{period_type}")
    filtered = table.loc[mask].reset_index(drop=True)

    st.caption(f"{len(filtered):,} of {len(table):,} products match the current filters.")
    if filtered.empty:
        st.info("No products match the current filters.")
        return

    metric_cols = st.columns(3)
    metric_cols[0].metric("Products", f"{len(filtered):,}")
    metric_cols[1].metric("Forecast demand (units)", f"{filtered['forecast'].sum():,.0f}")
    metric_cols[2].metric(
        "Recommended Target Inventory (units)", f"{filtered['target_inventory'].sum():,.0f}"
    )

    display = pd.DataFrame(
        {
            "Product": filtered["stock_code"],
            "Description": filtered["description"],
            "ABC": filtered["abc_class"],
            "Forecast": filtered["forecast"],
            "Safety Stock": filtered["safety_stock"],
            "Recommended Target Inventory": filtered["target_inventory"],
        }
    )
    if filtered["actual"].notna().any():
        display["Actual"] = filtered["actual"]
    if period_type == PERIOD_QUARTER:
        display["Months"] = filtered["months_included"]
        display["Complete"] = filtered["complete"]
    display["Basis"] = filtered["source"].map(lambda s: SOURCE_LABELS.get(s, s))

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Forecast": st.column_config.NumberColumn(format="%.1f"),
            "Safety Stock": st.column_config.NumberColumn(format="%.1f"),
            "Recommended Target Inventory": st.column_config.NumberColumn(format="%d"),
            "Actual": st.column_config.NumberColumn(format="%.0f"),
        },
    )

    run_id = data_access.load_run_log()["run_id"]
    st.download_button(
        "Download CSV",
        data=filtered.to_csv(index=False),
        file_name=f"period_plan_{selected_period}_{run_id}.csv",
        mime="text/csv",
    )

    if period_type == PERIOD_QUARTER:
        st.caption(
            "A quarter's Recommended Target Inventory is the sum of its monthly targets: with a "
            "one-month lead time the stock level is re-set every month, so the safety stock is "
            "held once per month, not once per quarter. There is no quarterly model."
        )
    st.caption(
        "Months beyond the operational one are recursive: each month's forecast becomes the next "
        "month's input, so accuracy degrades with distance. Every horizon carries its own sigma, "
        "measured from a back-test at that same horizon, so the safety stock widens accordingly."
    )


view = st.radio(
    "View",
    options=[VIEW_NEXT_MONTH, VIEW_BY_MONTH, VIEW_BY_QUARTER],
    horizontal=True,
    help="The operational plan for the next month, or the stocking requirement for any month or "
    "quarter the pipeline covers.",
)

if view == VIEW_NEXT_MONTH:
    render_operational_plan()
elif view == VIEW_BY_MONTH:
    render_period_view(PERIOD_MONTH)
else:
    render_period_view(PERIOD_QUARTER)
