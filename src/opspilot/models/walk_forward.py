"""v1.1: compare models month by month, each retrained on what was known that day.

Run with `make walk-forward`, after `make train`. For every month from July 2017
to the censoring cutoff, each model is fitted on orders whose outcome was known
on the month's first day, then scores that month. This is a broader comparison
than one split, not a clean test: features and parameters were chosen on data up
to March 2018, and the months from April 2018 are the v1 test set, already seen.

It also tests a calibration fix. v1's calibrator was fitted on one kind of model
and applied to another. Here a single number, a shift in log-odds, is learned from
the same model's own earlier predictions, and retrained models get it monthly.
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit

from opspilot.config import FOLD_STARTS, RANDOM_SEED, REPORTS_DIR, SPLITS
from opspilot.features.load import FEATURES, OUTCOME_KNOWN_AT, TARGET, load_features
from opspilot.models.baselines import (
    fit_logistic_regression,
    logistic_regression_on,
    route_rate_rule,
)
from opspilot.models.bundle import load_bundle
from opspilot.models.calibrate import SigmoidCalibrator
from opspilot.models.ensemble import AveragedModel
from opspilot.models.evaluate import AQUA, BLUE, GRAY, INK, ORANGE, _save, plt
from opspilot.models.metrics import brier, expected_calibration_error, lift, recall_top10, roc_auc
from opspilot.models.selection_report import markdown_table
from opspilot.models.split import known_before
from opspilot.models.tracking import git_commit
from opspilot.models.train import fit_xgboost, xgboost_fit_predict

OUT_DIR = REPORTS_DIR / "v1.1"
TEST_START = pd.Timestamp(SPLITS["test"][0])
HALF_LIFE_DAYS = 90
# The calibration level is learned from orders purchased 90 to 30 days before the
# month starts: recent, yet old enough for most outcomes to be known.
LEVEL_WINDOW_DAYS = (90, 30)
MIN_LEVEL_ROWS = 200
MODELS = [
    "route_rule",
    "logistic_regression",
    "xgboost",
    "average",
    "logistic_regression_recent",
    "xgboost_recent",
    "average_recent",
]
LABELS = {
    "route_rule": "Rule: route late rate",
    "logistic_regression": "Logistic regression",
    "xgboost": "XGBoost",
    "average": "Average of both",
    "logistic_regression_recent": "Logistic regression, recent weighted",
    "xgboost_recent": "XGBoost, recent weighted",
    "average_recent": "Average of both, recent weighted",
}


def months(end: pd.Timestamp | None = None) -> pd.PeriodIndex:
    return pd.period_range(FOLD_STARTS[0], end or SPLITS["test"][1], freq="M")


def average_log_odds(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return expit((logit(np.clip(a, 1e-6, 1 - 1e-6)) + logit(np.clip(b, 1e-6, 1 - 1e-6))) / 2)


def run(features: pd.DataFrame, params: dict) -> pd.DataFrame:
    """One row per order, month and model, with the raw out-of-sample prediction."""
    dates = pd.to_datetime(features.purchase_date)
    first_day = pd.Timestamp(SPLITS["train"][0])
    cutoff = pd.Timestamp(SPLITS["test"][1])
    frames = []
    for month in months():
        start, end = month.start_time, min(month.end_time.normalize(), cutoff)
        fit = features[(dates >= first_day) & (dates < start)]
        fit = fit[known_before(fit, start)]
        scored = features[dates.between(start, end)]
        p = {
            "route_rule": route_rate_rule(fit, scored),
            "logistic_regression": logistic_regression_on(FEATURES)(fit, scored),
            "xgboost": xgboost_fit_predict(params)(fit, scored),
            "logistic_regression_recent": logistic_regression_on(FEATURES, HALF_LIFE_DAYS)(
                fit, scored
            ),
            "xgboost_recent": xgboost_fit_predict(params, FEATURES, HALF_LIFE_DAYS)(fit, scored),
        }
        p["average"] = average_log_odds(p["logistic_regression"], p["xgboost"])
        p["average_recent"] = average_log_odds(p["logistic_regression_recent"], p["xgboost_recent"])
        base = scored[["order_id", "purchase_date", OUTCOME_KNOWN_AT, TARGET]].rename(
            columns={TARGET: "y"}
        )
        for name in MODELS:
            frames.append(base.assign(month=str(month), model=name, p=p[name]))
    return pd.concat(frames, ignore_index=True)


def level_shift(p: np.ndarray, y: np.ndarray) -> float:
    """The shift in log-odds that makes the mean prediction equal the observed late rate."""
    target = float(np.mean(y))
    if target in (0.0, 1.0):
        return 0.0
    logits = logit(np.clip(p, 1e-6, 1 - 1e-6))
    return float(brentq(lambda d: expit(logits + d).mean() - target, -15, 15))


def level_rows(history: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    """Earlier predictions usable on `start`: in the window, with the outcome known by then."""
    oldest, newest = (start - pd.Timedelta(days=d) for d in LEVEL_WINDOW_DAYS)
    dates = pd.to_datetime(history.purchase_date)
    return history[dates.between(oldest, newest, inclusive="left") & known_before(history, start)]


def with_corrections(predictions: pd.DataFrame) -> pd.DataFrame:
    """Add p_level and p_sigmoid, each fitted only on the model's own earlier predictions."""
    frames = []
    for _, model_rows in predictions.groupby("model", sort=False):
        for month, rows in model_rows.groupby("month", sort=True):
            start = pd.Period(month, freq="M").start_time
            usable = level_rows(model_rows[model_rows.month < month], start)
            rows = rows.copy()
            rows["p_level"], rows["p_sigmoid"], rows["shift"] = np.nan, np.nan, np.nan
            if len(usable) >= MIN_LEVEL_ROWS and 0 < usable.y.sum() < len(usable):
                shift = level_shift(usable.p.to_numpy(), usable.y.to_numpy())
                sigmoid = SigmoidCalibrator().fit(usable.p, usable.y)
                raw = logit(np.clip(rows.p.to_numpy(), 1e-6, 1 - 1e-6))
                rows["p_level"] = expit(raw + shift)
                rows["p_sigmoid"] = sigmoid.predict(rows.p)
                rows["shift"] = shift
            frames.append(rows)
    return pd.concat(frames, ignore_index=True)


def _period(month: str) -> str:
    return "from April 2018 (v1 test)" if pd.Timestamp(month) >= TEST_START else "before April 2018"


def monthly_ranking(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, month), part in predictions.groupby(["model", "month"], sort=False):
        y, p = part.y.to_numpy(), part.p.to_numpy()
        if not 0 < y.sum() < len(y):
            continue
        rows.append(
            {
                "model": model,
                "month": month,
                "orders": len(y),
                "late rate": y.mean(),
                "lift": lift(y, p),
                "roc_auc": roc_auc(y, p),
                "recall_top10": recall_top10(y, p),
            }
        )
    return pd.DataFrame(rows)


def ranking_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        part = monthly[monthly.model == model]
        recent = part[part.month.map(_period).str.startswith("from")]
        rows.append(
            {
                "Model": LABELS[model],
                "Mean lift": part.lift.mean(),
                "Mean ROC-AUC": part.roc_auc.mean(),
                "Top-10% recall": part.recall_top10.mean(),
                "Top-10% recall, Apr to Jul 2018": recent.recall_top10.mean(),
            }
        )
    return pd.DataFrame(rows)


def paired_months(monthly: pd.DataFrame, a: str, b: str, metric: str) -> dict:
    """Month-by-month difference a minus b, with an interval from resampling months."""
    wide = monthly.pivot(index="month", columns="model", values=metric)
    difference = (wide[a] - wide[b]).dropna().to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)
    means = [rng.choice(difference, len(difference)).mean() for _ in range(10_000)]
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "Comparison": f"{LABELS[a]} minus {LABELS[b]}",
        "Metric": metric,
        "Mean difference": difference.mean(),
        "95% low": low,
        "95% high": high,
        "Months ahead": f"{int((difference > 0).sum())} of {len(difference)}",
    }


def calibration_summary(predictions: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Raw against corrected, over the months where a correction existed.

    Months before April 2018 are shown apart: a choice can rest on them without
    leaning on the v1 test months.
    """
    rows = []
    usable = predictions[predictions.p_level.notna()]
    usable = usable.assign(period=usable.month.map(_period))
    groups = usable.groupby(["period", "model"], sort=True)
    for (period, model), part in groups:
        if model not in models:
            continue
        for label, column in [("raw", "p"), ("level shift", "p_level"), ("sigmoid", "p_sigmoid")]:
            by_month = part.groupby("month").agg(mean_p=(column, "mean"), late=("y", "mean"))
            rows.append(
                {
                    "Months": period,
                    "Model": LABELS[model],
                    "Correction": label,
                    "Mean gap, predicted vs actual": (by_month.mean_p - by_month.late).abs().mean(),
                    "Brier": brier(part.y, part[column]),
                    "ECE": expected_calibration_error(part.y, part[column]),
                }
            )
    return pd.DataFrame(rows)


def band_cutoffs(scores: np.ndarray) -> dict[str, float]:
    """High is the top 10% of recent scores, medium the next 20%."""
    return {"high": float(np.quantile(scores, 0.9)), "medium": float(np.quantile(scores, 0.7))}


def fit_rows(features: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    dates = pd.to_datetime(features.purchase_date)
    rows = features[(dates >= pd.Timestamp(SPLITS["train"][0])) & (dates < as_of)]
    return rows[known_before(rows, as_of)]


def fit_recipe(features: pd.DataFrame, params: dict, as_of: pd.Timestamp) -> AveragedModel:
    """The v1.1 model as of a date: both models on outcomes known by then, no calibrator."""
    fit = fit_rows(features, as_of)
    logistic = fit_logistic_regression(fit)
    reference = logistic[0].transform(fit[FEATURES]).mean().to_numpy()
    return AveragedModel(logistic, fit_xgboost(fit, params), reference)


def cutoffs_for(model: AveragedModel, features: pd.DataFrame, as_of: pd.Timestamp) -> dict | None:
    """Rank the last 90 days of orders with this same model. This needs no outcomes."""
    dates = pd.to_datetime(features.purchase_date)
    recent = features[dates.between(as_of - pd.Timedelta(days=90), as_of, inclusive="left")]
    if len(recent) < MIN_LEVEL_ROWS:
        return None
    return band_cutoffs(model.predict_proba(recent[FEATURES])[:, 1])


def band_check(features: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Each month: fit the recipe on its first day, set cut-offs, apply them to that month."""
    dates = pd.to_datetime(features.purchase_date)
    cutoff = pd.Timestamp(SPLITS["test"][1])
    table = []
    for month in months():
        start, end = month.start_time, min(month.end_time.normalize(), cutoff)
        model = fit_recipe(features, params, start)
        cuts = cutoffs_for(model, features, start)
        if cuts is None:
            continue
        part = features[dates.between(start, end)]
        p, y = model.predict_proba(part[FEATURES])[:, 1], part[TARGET].to_numpy()
        high, medium = p >= cuts["high"], (p >= cuts["medium"]) & (p < cuts["high"])
        low = p < cuts["medium"]
        table.append(
            {
                "month": str(month),
                "late rate": y.mean(),
                "share high": high.mean(),
                "late in high": y[high].mean() if high.any() else np.nan,
                "share medium": medium.mean(),
                "late in medium": y[medium].mean() if medium.any() else np.nan,
                "late in low": y[low].mean() if low.any() else np.nan,
            }
        )
    return pd.DataFrame(table)


def level_by_month(predictions: pd.DataFrame, model: str) -> pd.DataFrame:
    part = predictions[predictions.model == model]
    return part.groupby("month").agg(
        late_rate=("y", "mean"), raw=("p", "mean"), level_shift=("p_level", "mean")
    )


def figure_recall(monthly: pd.DataFrame, path) -> None:
    wide = monthly.pivot(index="month", columns="model", values="recall_top10")
    fig, ax = plt.subplots(figsize=(7.4, 3.8), layout="constrained")
    for model, color in [("xgboost", BLUE), ("logistic_regression", ORANGE), ("route_rule", AQUA)]:
        ax.plot(
            wide.index,
            wide[model],
            color=color,
            linewidth=2,
            marker="o",
            markersize=4,
            label=LABELS[model],
        )
    ax.axhline(0.10, color=GRAY, linewidth=1, linestyle=(0, (4, 3)), label="Random")
    ax.set(ylabel="Share of late orders in riskiest 10%", ylim=(0, None))
    ax.tick_params(axis="x", rotation=45)
    ax.set_title("Each model retrained monthly")
    ax.legend(loc="upper left", ncols=2)
    _save(fig, path)


def figure_level(levels: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.8), layout="constrained")
    ax.plot(
        levels.index,
        levels.late_rate,
        color=INK,
        linewidth=2,
        marker="o",
        markersize=4,
        label="Actual late rate",
    )
    ax.plot(levels.index, levels.raw, color=ORANGE, linewidth=2, label="Retrained model, raw")
    ax.plot(levels.index, levels.level_shift, color=BLUE, linewidth=2, label="With level shift")
    ax.set(ylabel="Share of orders", ylim=(0, None))
    ax.tick_params(axis="x", rotation=45)
    ax.set_title("Predicted against actual late rate, month by month")
    ax.legend(loc="upper left")
    _save(fig, path)


def write_report(predictions: pd.DataFrame, bands: pd.DataFrame) -> None:
    monthly = monthly_ranking(predictions)
    months_shown = monthly[monthly.model.isin(["xgboost", "logistic_regression", "route_rule"])]
    recall = months_shown.pivot(index="month", columns="model", values="recall_top10")
    recall.insert(0, "late rate", monthly.groupby("month")["late rate"].first())
    recall.insert(0, "orders", monthly.groupby("month")["orders"].first())
    comparisons = pd.DataFrame(
        [
            paired_months(monthly, "xgboost", "logistic_regression", metric)
            for metric in ["lift", "roc_auc", "recall_top10"]
        ]
        + [
            paired_months(monthly, "average", "xgboost", "lift"),
            paired_months(monthly, "average", "logistic_regression", "lift"),
            paired_months(monthly, "average", "xgboost", "roc_auc"),
            paired_months(monthly, "average", "logistic_regression", "roc_auc"),
            paired_months(monthly, "average", "xgboost", "recall_top10"),
            paired_months(monthly, "average", "logistic_regression", "recall_top10"),
            paired_months(monthly, "xgboost_recent", "xgboost", "recall_top10"),
            paired_months(
                monthly, "logistic_regression_recent", "logistic_regression", "recall_top10"
            ),
        ]
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_recall(monthly, OUT_DIR / "figures" / "recall_by_month.png")
    levels = level_by_month(predictions, "average")
    figure_level(levels, OUT_DIR / "figures" / "level_by_month.png")
    sections = [
        "# v1.1: month-by-month comparison",
        f"Generated by `make walk-forward` at commit `{git_commit()[:7]}`; do not edit by hand. "
        "Every model is retrained each month on orders whose outcome was known on the month's "
        "first day. **This is not a clean test:** features and parameters were chosen on data up "
        "to March 2018, and the months from April 2018 are the v1 test set, already seen. The "
        "frozen v1 model is not changed by any of this.",
        "## Ranking across all months",
        markdown_table(ranking_summary(monthly)),
        "## Late orders caught in the riskiest 10%, by month",
        "![Recall by month](figures/recall_by_month.png)",
        markdown_table(recall.rename_axis("month"), digits=3),
        "## Are the differences real?",
        "Each row is one model minus another, month by month. The interval comes from "
        "resampling the months, of which there are few.",
        markdown_table(comparisons),
        "## Calibration fix",
        "Months where a correction was available. A level shift moves all of a model's "
        "predictions by one number, learned from its own earlier predictions on orders "
        "purchased 90 to 30 days before the month starts, using outcomes known by then.",
        markdown_table(
            calibration_summary(predictions, ["xgboost", "logistic_regression", "average"]),
            digits=4,
        ),
        "![Level by month](figures/level_by_month.png)",
        markdown_table(levels.rename_axis("month"), digits=3),
        "## Risk bands",
        "The served recipe is the average of both models, retrained monthly, with no "
        "calibrator. On its first day each month's model scores the previous 90 days of orders; "
        "high is their top 10%, medium the next 20%. No outcomes are needed.",
        markdown_table(bands),
    ]
    (OUT_DIR / "walk_forward.md").write_text("\n\n".join(sections) + "\n")


def main() -> None:
    features, params = load_features(), load_bundle()["params"]
    write_report(with_corrections(run(features, params)), band_check(features, params))
    print(f"Wrote {OUT_DIR / 'walk_forward.md'}")


if __name__ == "__main__":
    main()
