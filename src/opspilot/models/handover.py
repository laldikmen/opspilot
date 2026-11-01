"""v1.1: how much better is a second prediction at carrier handover?

Run with `make handover`, after `make features train`. Each month both versions of
the recipe are fitted on what was known on the month's first day, then score the
same orders: those still open at handover. Orders handed over after their promised
day are left out, because they are already known to be late; counting them would
flatter the result. Like the rest of v1.1, this is not a clean test.
"""

import numpy as np
import pandas as pd

from opspilot.config import RANDOM_SEED, SPLITS
from opspilot.features.load import (
    FEATURES,
    HANDOVER_FEATURES,
    TARGET,
    load_features,
    load_handover_features,
)
from opspilot.models.baselines import logistic_regression_on
from opspilot.models.bundle import load_bundle
from opspilot.models.metrics import lift, precision_recall_at_top, recall_top10, roc_auc
from opspilot.models.selection_report import markdown_table
from opspilot.models.split import known_before
from opspilot.models.tracking import git_commit
from opspilot.models.train import xgboost_fit_predict
from opspilot.models.walk_forward import (
    OUT_DIR,
    average_log_odds,
    fit_rows,
    months,
)

ALL_FEATURES = FEATURES + HANDOVER_FEATURES
MODELS = {
    "at_approval": "Predicted at approval",
    "at_handover": "Predicted at handover",
    "delay_rule": "Rule: seller later than usual",
}


def average(fit: pd.DataFrame, scored: pd.DataFrame, features: list[str], params: dict):
    logistic = logistic_regression_on(features)(fit, scored)
    xgboost = xgboost_fit_predict(params, features)(fit, scored)
    return average_log_odds(logistic, xgboost)


def run(features: pd.DataFrame, handover: pd.DataFrame, params: dict) -> pd.DataFrame:
    """One row per open order: its month, label, and the score from each moment."""
    open_orders = features.merge(handover, on="order_id")
    dates = pd.to_datetime(open_orders.purchase_date)
    cutoff = pd.Timestamp(SPLITS["test"][1])
    frames = []
    for month in months():
        start, end = month.start_time, min(month.end_time.normalize(), cutoff)
        scored = open_orders[dates.between(start, end)]
        fit_approval = fit_rows(features, start)
        fit_handover = fit_rows(open_orders, start)
        frames.append(
            pd.DataFrame(
                {
                    "month": str(month),
                    "y": scored[TARGET].to_numpy(),
                    "days_left": scored.days_left.to_numpy(),
                    "at_approval": average(fit_approval, scored, FEATURES, params),
                    "at_handover": average(fit_handover, scored, ALL_FEATURES, params),
                    "delay_rule": scored.handover_vs_usual_days.fillna(0).to_numpy(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def monthly_table(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for month, part in scores.groupby("month", sort=True):
        y = part.y.to_numpy()
        if not 0 < y.sum() < len(y):
            continue
        for name in MODELS:
            p = part[name].to_numpy()
            rows.append(
                {
                    "month": month,
                    "model": name,
                    "orders": len(y),
                    "late rate": y.mean(),
                    "lift": lift(y, p),
                    "roc_auc": roc_auc(y, p),
                    "recall_top10": recall_top10(y, p),
                }
            )
    return pd.DataFrame(rows)


def summary(monthly: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, label in MODELS.items():
        part = monthly[monthly.model == name]
        rows.append(
            {
                "Model": label,
                "Mean lift": part.lift.mean(),
                "Mean ROC-AUC": part.roc_auc.mean(),
                "Top-10% recall": part.recall_top10.mean(),
                "Top-20% recall": np.mean(
                    [
                        precision_recall_at_top(p.y, p[name], 0.2)[1]
                        for _, p in scores.groupby("month")
                        if p.y.sum() > 0
                    ]
                ),
            }
        )
    return pd.DataFrame(rows)


def by_days_left(scores: pd.DataFrame) -> pd.DataFrame:
    bands = pd.cut(scores.days_left, [-2, 2, 5, 10, 15, 30, 200])
    table = scores.groupby(bands, observed=True).y.agg(["size", "mean"]).reset_index()
    table.columns = ["Days left", "Orders", "Late rate"]
    table["Days left"] = table["Days left"].astype(str)
    return table


def scrambled(handover: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Handover features shuffled within each month: any real gain should disappear."""
    months_of = handover.order_id.map(
        features.set_index("order_id").purchase_date.astype(str).str[:7]
    )
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = handover.copy()
    shuffled[HANDOVER_FEATURES] = handover.groupby(months_of)[HANDOVER_FEATURES].transform(
        lambda column: rng.permutation(column.to_numpy())
    )
    return shuffled


def write_report(
    scores: pd.DataFrame, check: pd.DataFrame, features: pd.DataFrame, handover: pd.DataFrame
) -> None:
    monthly = monthly_table(scores)
    wide = monthly.pivot(index="month", columns="model", values="recall_top10")
    first = monthly.groupby("month")[["orders", "late rate"]].first()
    wide = first.join(wide)[["orders", "late rate", *MODELS]]
    known = features[known_before(features, pd.Timestamp("2100-01-01"))]
    window = known[pd.to_datetime(known.purchase_date) >= pd.Timestamp(str(months()[0]))]
    missing = window[~window.order_id.isin(handover.order_id)]
    late_covered = window[TARGET].sum()
    flagged = scores[
        scores.at_handover
        >= scores.groupby("month").at_handover.transform(lambda s: s.quantile(0.9))
    ]
    sections = [
        "# v1.1: a second prediction at carrier handover",
        f"Generated by `make handover` at commit `{git_commit()[:7]}`; do not edit by hand. Both "
        "predictions use the v1.1 recipe, fitted each month on outcomes known on its first day, "
        "and score the same orders. **Not a clean test:** the April to July months are the v1 "
        "test set, already seen.",
        "## Result",
        f"{len(scores):,} orders were still open at handover, {scores.y.mean():.2%} of them "
        "eventually late.",
        markdown_table(summary(monthly, scores)),
        "Lift jumps mostly because of a few near-certain orders, handed over with almost no "
        "time left, so top-10% recall is the fairer headline. By days left at handover, the "
        "late rate is:",
        markdown_table(by_days_left(scores)),
        "## Leak check",
        "With the handover features shuffled within each month the gain should vanish, because "
        "they are the only thing that differs between the two predictions.",
        markdown_table(summary(monthly_table(check), check)),
        "## Late orders caught in the riskiest 10%, by month",
        markdown_table(wide.rename_axis("month").reset_index()),
        "## What the scores leave out",
        f"Of {len(window):,} orders in these months, {len(missing):,} are not scored at handover, "
        f"and they hold {missing[TARGET].sum():,} of the {late_covered:,} late orders. They were "
        "handed over after their promised day, or have no handover record, or were delivered "
        "first, so a late order among them needs no model: the missed promise is already visible.",
        f"Flagged orders at handover have a median of {flagged.days_left.median():.1f} days left "
        "before the promised date: that is the time an operations team has to act.",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "handover.md").write_text("\n\n".join(sections) + "\n")


def main() -> None:
    features, handover = load_features(), load_handover_features()
    params = load_bundle()["params"]
    scores = run(features, handover, params)
    check = run(features, scrambled(handover, features), params)
    write_report(scores, check, features, handover)
    print(f"Wrote {OUT_DIR / 'handover.md'}")


if __name__ == "__main__":
    main()
