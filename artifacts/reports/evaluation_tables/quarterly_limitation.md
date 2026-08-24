# Quarterly forecast — methodology limitation

This project trains and evaluates a single **one-step-ahead monthly model**: at any forecast
origin, it predicts only the single month that immediately follows. Every quarterly figure in
`quarterly_forecast.csv` is a **sum of three such one-step-ahead forecasts**, each produced at its
own origin (the month before the one it predicts) by the rolling back-test — never a separate
quarterly model, and never a single forecast covering all three months at once.

As a direct consequence, this module
**cannot forecast all three months of a quarter at the start of the quarter**.
The second and third months of any quarter appear here only once the preceding month's data
becomes available, one month at a time.

A genuine start-of-quarter, multi-month forecast is produced by a *separate* module,
`pipeline.multi_horizon`: it runs the same one-step-ahead champion **recursively**, feeding each
month's forecast back into the panel as that month's units before predicting the next, and measures
a distinct sigma for every horizon from a back-test run at that same horizon. Its output is
`multi_horizon_plan.csv` and `period_plan.csv`, and its numbers are *not* interchangeable with the
ones in this file: a later horizon compounds the error of every horizon before it, which is exactly
why it carries its own, wider safety stock. `quarterly_forecast.csv` remains what it says it is —
sums of genuine one-step-ahead forecasts, each made one month before its target.

The rolling estimate for the current partial quarter is not an exception to this: it combines the
already-observed actual sales of the quarter's completed months with the single genuine
one-step-ahead forecast for the next month, and is therefore never treated as a scored, complete
quarter (`complete = False`, no `actual_sum`).
