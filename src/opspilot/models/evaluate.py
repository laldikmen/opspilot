"""Test-set report: results table, slices, figures.

Run once with `make evaluate`. The test set is scored once and the model is not
changed afterwards (docs/decisions/0004-chronological-split.md). This loads the
frozen bundle and calibrators saved by `make train`, fits the baselines on train,
scores every model on test, and writes reports/evaluation.md, the figures in
reports/figures/ and artifacts/test_predictions.csv.

`--split validation` runs the same report on validation, which model selection
already used, into artifacts/dry_run/. That dry run is how this code was checked
before the test set was scored.
"""

import argparse
import hashlib
import re
from datetime import date
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from opspilot.config import ARTIFACTS_DIR, BUNDLE_PATH, REPORTS_DIR, SPLITS  # noqa: E402
from opspilot.features.load import FEATURES, TARGET, load_features  # noqa: E402
from opspilot.models.baselines import BASELINES  # noqa: E402
from opspilot.models.bundle import load_bundle  # noqa: E402
from opspilot.models.explain import global_importance, top_reasons  # noqa: E402
from opspilot.models.metrics import (  # noqa: E402
    bootstrap_interval,
    evaluate,
    lift,
    paired_bootstrap,
    pr_auc,
    precision_recall_at_top,
    recall_top10,
    roc_auc,
)
from opspilot.models.selection_report import markdown_table  # noqa: E402
from opspilot.models.split import Period, final_period, known_before  # noqa: E402
from opspilot.models.tracking import git_commit  # noqa: E402

CALIBRATORS_PATH = ARTIFACTS_DIR / "calibrators.joblib"
PROTOCOL_PATH = REPORTS_DIR / "test_protocol.md"

# Fixed in reports/test_protocol.md before the test set was scored: for each
# calibration source, the method with the lower backtest Brier score (ADR 0006).
CALIBRATED = {
    "xgboost_calibrated_validation": "last_isotonic",
    "xgboost_calibrated_pooled": "pooled_sigmoid",
}
FINAL = "xgboost_calibrated_pooled"
LABELS = {
    "prior": "Prior (train late rate)",
    "seller_rate_rule": "Rule: seller smoothed late rate",
    "route_rate_rule": "Rule: route late rate",
    "logistic_regression": "Logistic regression",
    "xgboost_uncalibrated": "XGBoost, uncalibrated",
    "xgboost_calibrated_validation": "XGBoost, calibrated on validation",
    "xgboost_calibrated_pooled": "XGBoost, calibrated on pooled folds (final)",
}
RIVALS = ["logistic_regression", "route_rate_rule"]
MIN_SLICE = 200
CAPACITY_SHARES = [0.05, 0.10, 0.20, 0.30]
TOP_CATEGORIES = 10

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8c8b85"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3de", "#fcfcfb"
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.titlelocation": "left",
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "font.size": 9,
        "legend.frameon": False,
    }
)


# --- Scoring -----------------------------------------------------------------


def period_for(split: str) -> Period:
    if split == "test":
        return final_period()
    train_start, train_end = SPLITS["train"]
    start, end = SPLITS["validation"]
    return Period("validation", train_start, train_end, start, end)


def score(features: pd.DataFrame, split: str, bundle: dict, calibrators: dict) -> pd.DataFrame:
    """Every model's prediction for every order in the split, with the label as y."""
    fit_rows, score_rows = period_for(split).masks(features)
    fit, scored = features[fit_rows], features[score_rows]
    scores = scored[["order_id", "purchase_date", "customer_state", "main_category", TARGET]]
    scores = scores.rename(columns={TARGET: "y"})
    for name, fit_predict in BASELINES.items():
        scores[name] = fit_predict(fit, scored)
    raw = bundle["model"].predict_proba(scored[bundle["features"]])[:, 1]
    scores["xgboost_uncalibrated"] = raw
    for name, strategy in CALIBRATED.items():
        scores[name] = calibrators[strategy].predict(raw)
    return scores


def prediction_fingerprint(bundle: dict, features: pd.DataFrame) -> str:
    """A hash of the bundle's raw predictions on validation.

    Unlike the file hash, it survives rebuilding the bundle on another day, since
    training is deterministic and only the version string would change.
    """
    _, rows = period_for("validation").masks(features)
    raw = bundle["model"].predict_proba(features.loc[rows, bundle["features"]])[:, 1]
    return hashlib.sha256(np.round(raw, 8).tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "no file"


def protocol_fingerprint(path: Path = PROTOCOL_PATH) -> str | None:
    """The prediction fingerprint recorded in the protocol before test was scored."""
    if not path.exists():
        return None
    found = re.search(r"Prediction fingerprint: `([0-9a-f]{64})`", path.read_text())
    return found.group(1) if found else None


# --- Tables ------------------------------------------------------------------


def _with_interval(value: float, interval: tuple[float, float]) -> str:
    return f"{value:.3f} ({interval[0]:.3f} to {interval[1]:.3f})"


def results_table(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, label in LABELS.items():
        y, p = scores.y.to_numpy(), scores[name].to_numpy()
        m = evaluate(y, p)
        rows.append(
            {
                "Model": label,
                "PR-AUC (95% CI)": _with_interval(m["pr_auc"], bootstrap_interval(y, p, pr_auc)),
                "Lift": m["lift"],
                "ROC-AUC": m["roc_auc"],
                "Precision @ top 10%": m["precision_top10"],
                "Recall @ top 10% (95% CI)": _with_interval(
                    m["recall_top10"], bootstrap_interval(y, p, recall_top10)
                ),
                "Brier": m["brier"],
                "ECE": m["ece"],
                "Mean prediction": float(np.mean(p)),
            }
        )
    return pd.DataFrame(rows)


def comparison_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Paired bootstraps of the final model against the strongest baselines."""
    rows = []
    for rival in RIVALS:
        args = (scores.y, scores[FINAL], scores[rival])
        pr = paired_bootstrap(*args, metric=pr_auc)
        recall = paired_bootstrap(*args, metric=recall_top10)
        rows.append(
            {
                "Final model compared with": LABELS[rival],
                "PR-AUC difference (95% CI)": _with_interval(pr[0], pr[1:]),
                "Top-10% recall difference (95% CI)": _with_interval(recall[0], recall[1:]),
            }
        )
    return pd.DataFrame(rows)


def capacity_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Share of late orders caught when the riskiest X% of orders are checked."""
    rows = []
    for name in [FINAL, "logistic_regression", "route_rate_rule", "prior"]:
        rows.append(
            {
                "Model": LABELS[name],
                **{
                    f"Check {share:.0%}": precision_recall_at_top(scores.y, scores[name], share)[1]
                    for share in CAPACITY_SHARES
                },
            }
        )
    return pd.DataFrame(rows)


def slice_table(scores: pd.DataFrame, column: str, keep: list | None = None) -> pd.DataFrame:
    """The final model by slice. Recall uses the global top-10% cut-off, as an ops team would."""
    threshold = np.quantile(scores[FINAL], 0.9)
    rows = []
    for value, part in scores.groupby(column, dropna=False):
        if keep is not None and value not in keep:
            continue
        y, p = part.y.to_numpy(), part[FINAL].to_numpy()
        late = int(y.sum())
        rows.append(
            {
                column: value if isinstance(value, str) else "missing",
                "Orders": len(part),
                "Late rate": y.mean(),
                "Mean prediction": p.mean(),
                "ROC-AUC": roc_auc(y, p) if 0 < late < len(y) else np.nan,
                "Lift": lift(y, p) if late else np.nan,
                "Late orders in global top 10%": (p[y == 1] >= threshold).mean()
                if late
                else np.nan,
                "Reliable": "yes" if len(part) >= MIN_SLICE else "no, under 200 orders",
            }
        )
    return pd.DataFrame(rows).sort_values("Orders", ascending=False)


def example_explanations(scores: pd.DataFrame, rows: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Top three reasons for three telling orders, picked by a fixed rule."""
    final = scores[FINAL]
    picks = {
        "Riskiest order that was late": final[scores.y == 1].idxmax(),
        "Riskiest order that arrived on time": final[scores.y == 0].idxmax(),
        "Late order the model rated safest": final[scores.y == 1].idxmin(),
    }
    out = []
    for label, index in picks.items():
        reasons = top_reasons(bundle, rows.loc[[index]])[0]
        out.append(
            {
                "Order": label,
                "Predicted": final[index],
                "Actual": "late" if scores.at[index, "y"] else "on time",
                **{f"Reason {i}": _reason(r) for i, r in enumerate(reasons, start=1)},
            }
        )
    return pd.DataFrame(out)


def _reason(reason: dict) -> str:
    return f"{reason['feature']} = {_short(reason['value'])} ({reason['contribution']:+.2f})"


def _short(value) -> str:
    if value is None:
        return "missing"
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def monthly_backtest(features: pd.DataFrame, split: str, bundle: dict) -> pd.DataFrame:
    """Supplementary: retrain each month on everything known by its first day, score that month.

    Answers "what if the model were retrained monthly?" next to the frozen model.
    Not used to change the frozen model.
    """
    from opspilot.models.train import fit_xgboost

    start, end = (pd.Timestamp(d) for d in SPLITS[split])
    train_start = pd.Timestamp(SPLITS["train"][0])
    dates = pd.to_datetime(features.purchase_date)
    rows = []
    for month in pd.period_range(start, end, freq="M"):
        month_start, month_end = month.start_time, min(month.end_time.normalize(), end)
        fit = features[(dates >= train_start) & (dates < month_start)]
        fit = fit[known_before(fit, month_start)]
        scored = features[dates.between(month_start, month_end)]
        retrained = fit_xgboost(fit, bundle["params"]).predict_proba(scored[FEATURES])[:, 1]
        frozen = bundle["model"].predict_proba(scored[bundle["features"]])[:, 1]
        y = scored[TARGET].to_numpy()
        for name, p in [("frozen", frozen), ("retrained", retrained)]:
            rows.append(
                {
                    "Month": str(month),
                    "Model": name,
                    "Orders": len(scored),
                    "Late rate": y.mean(),
                    "Mean prediction": p.mean(),
                    "Lift": lift(y, p),
                    "ROC-AUC": roc_auc(y, p),
                    "Top-10% recall": recall_top10(y, p),
                }
            )
    return pd.DataFrame(rows)


# --- Figures -----------------------------------------------------------------


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_pr_curves(scores: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 4.2), layout="constrained")
    for name, color in [(FINAL, BLUE), ("logistic_regression", ORANGE), ("route_rate_rule", AQUA)]:
        precision, recall, _ = precision_recall_curve(scores.y, scores[name])
        ax.plot(
            recall,
            precision,
            color=color,
            linewidth=2,
            label=f"{LABELS[name]} (PR-AUC {pr_auc(scores.y, scores[name]):.3f})",
        )
    ax.axhline(
        scores.y.mean(),
        color=GRAY,
        linewidth=1,
        linestyle=(0, (4, 3)),
        label=f"Random (late rate {scores.y.mean():.3f})",
    )
    ax.set(
        xlabel="Recall: share of late orders flagged",
        ylabel="Precision: share of flags that are late",
        xlim=(0, 1),
        ylim=(0, None),
    )
    ax.set_title("Precision and recall")
    ax.legend(loc="upper right")
    return _save(fig, path)


def figure_capacity(scores: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 4.2), layout="constrained")
    for name, color in [(FINAL, BLUE), ("logistic_regression", ORANGE), ("route_rate_rule", AQUA)]:
        order = np.argsort(-scores[name].to_numpy(), kind="stable")
        caught = np.cumsum(scores.y.to_numpy()[order]) / scores.y.sum()
        checked = np.arange(1, len(order) + 1) / len(order)
        ax.plot(checked, caught, color=color, linewidth=2, label=LABELS[name])
    ax.plot([0, 1], [0, 1], color=GRAY, linewidth=1, linestyle=(0, (4, 3)), label="Random")
    ax.axvline(0.10, color=INK, linewidth=0.8, linestyle=(0, (2, 2)))
    ax.annotate(
        "an ops team checks 1 in 10", (0.10, 0.02), xytext=(4, 0), textcoords="offset points"
    )
    ax.set(
        xlabel="Share of orders checked, riskiest first",
        ylabel="Share of late orders caught",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.set_title("How many late orders a review team catches")
    ax.legend(loc="lower right")
    return _save(fig, path)


def _reliability(y, p, bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    groups = pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False)
    frame = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p), "g": groups.to_numpy()})
    means = frame.groupby("g")[["p", "y"]].mean()
    return means.p.to_numpy(), means.y.to_numpy()


def figure_calibration(
    scores: pd.DataFrame, validation: pd.DataFrame, split: str, path: Path
) -> Path:
    fig, ax = plt.subplots(figsize=(5.4, 4.6), layout="constrained")
    top = max(scores[FINAL].max(), validation[FINAL].max(), scores.y.mean(), validation.y.mean())
    ax.plot(
        [0, top],
        [0, top],
        color=GRAY,
        linewidth=1,
        linestyle=(0, (4, 3)),
        label="Perfect calibration",
    )
    series = [(validation, ORANGE, f"Validation (late rate {validation.y.mean():.1%})")]
    if split == "test":
        series.append((scores, BLUE, f"Test (late rate {scores.y.mean():.1%})"))
    for frame, color, label in series:
        x, y = _reliability(frame.y, frame[FINAL])
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=5, label=label)
    ax.set(
        xlabel="Predicted late probability (decile mean)",
        ylabel="Observed late rate",
        xlim=(0, None),
        ylim=(0, None),
    )
    ax.set_title("Calibration of the final model")
    ax.legend(loc="upper left")
    return _save(fig, path)


def figure_risk_deciles(scores: pd.DataFrame, path: Path) -> Path:
    deciles = pd.qcut(scores[FINAL].rank(method="first"), 10, labels=range(1, 11))
    rates = scores.groupby(deciles, observed=True).y.mean()
    fig, ax = plt.subplots(figsize=(6.4, 3.6), layout="constrained")
    ax.bar(rates.index.astype(int), rates.to_numpy(), color=BLUE, width=0.7)
    ax.axhline(scores.y.mean(), color=INK, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate(
        f"overall {scores.y.mean():.1%}",
        (0.6, scores.y.mean()),
        xytext=(0, 4),
        textcoords="offset points",
    )
    for decile, rate in rates.items():
        ax.annotate(
            f"{rate:.1%}",
            (int(decile), rate),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=MUTED,
        )
    ax.set(xlabel="Predicted risk decile (10 = riskiest)", ylabel="Observed late rate")
    ax.set_xticks(range(1, 11))
    ax.grid(axis="x", visible=False)
    ax.set_title("Late rate by predicted risk decile")
    return _save(fig, path)


def figure_importance(importance: pd.Series, path: Path, top: int = 15) -> Path:
    shown = importance.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.6), layout="constrained")
    ax.barh(shown.index, shown.to_numpy(), color=BLUE, height=0.65)
    ax.set(xlabel="Mean |SHAP contribution| (log-odds)")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"What drives the predictions: top {top} features")
    return _save(fig, path)


# --- Report ------------------------------------------------------------------


def write_report(
    split: str,
    scores: pd.DataFrame,
    validation: pd.DataFrame,
    rows: pd.DataFrame,
    features: pd.DataFrame,
    bundle: dict,
    out_dir: Path,
) -> Path:
    figures = out_dir / "figures"
    figure_pr_curves(scores, figures / "pr_curves.png")
    figure_capacity(scores, figures / "capacity.png")
    figure_calibration(scores, validation, split, figures / "calibration.png")
    figure_risk_deciles(scores, figures / "risk_deciles.png")
    figure_importance(global_importance(bundle, rows), figures / "shap_importance.png")

    fingerprint = prediction_fingerprint(bundle, features)
    registered = protocol_fingerprint()
    match = "yes" if registered == fingerprint else "no" if registered else "no protocol recorded"
    start, end = SPLITS[split]
    top_categories = scores.main_category.value_counts().head(TOP_CATEGORIES).index.tolist()
    month = pd.to_datetime(scores.purchase_date).dt.strftime("%Y-%m")
    final = evaluate(scores.y, scores[FINAL])

    title = "Evaluation" if split == "test" else "Evaluation dry run on validation"
    sections = [
        f"# {title}",
        f"Generated by `make evaluate{'' if split == 'test' else ' --split validation'}` on "
        f"{date.today():%Y-%m-%d} at commit `{git_commit()[:7]}`; do not edit by hand. "
        f"The {split} set ({start} to {end}, {len(scores):,} orders, late rate "
        f"{scores.y.mean():.2%}) is scored by the frozen bundle `{bundle['version']}`. "
        "Baselines are fitted on train only.",
        f"- Prediction fingerprint: `{fingerprint}`\n"
        f"- Matches the fingerprint in `reports/test_protocol.md`: **{match}**\n"
        f"- Bundle file SHA-256: `{file_sha256(BUNDLE_PATH)}`",
        "## Results",
        "Lift is PR-AUC divided by the late rate; a random model scores 1. Intervals come from "
        "1,000 bootstrap resamples of the scored orders.",
        markdown_table(results_table(scores)),
        "## The final model against the strongest baselines",
        "Paired bootstrap: the same resampled orders for both models.",
        markdown_table(comparison_table(scores)),
        "## Capacity: how many late orders a review team catches",
        f"If an operations team can check the riskiest 10% of orders, the final model points it "
        f"to {final['recall_top10']:.1%} of all late orders, and {final['precision_top10']:.1%} "
        "of the orders it checks turn out late.",
        markdown_table(capacity_table(scores)),
        "![Capacity curve](figures/capacity.png)",
        "![Precision and recall](figures/pr_curves.png)",
        "## Calibration",
        "Validation was part of the final calibrator's fitting data; the "
        f"{split} curve is the out-of-time check.",
        "![Calibration](figures/calibration.png)",
        "![Risk deciles](figures/risk_deciles.png)",
        "## Slices",
        f"The final model by slice. Slices with fewer than {MIN_SLICE} orders are flagged as "
        "unreliable. The recall column uses the global top-10% cut-off.",
        "### By customer state",
        markdown_table(slice_table(scores, "customer_state")),
        f"### Top {TOP_CATEGORIES} product categories",
        markdown_table(slice_table(scores, "main_category", keep=top_categories)),
        "### By purchase month",
        markdown_table(slice_table(scores.assign(month=month), "month")),
        "## Explanations",
        "SHAP contributions of the uncalibrated model, in log-odds; one-hot columns are summed "
        "back into their feature. A positive contribution pushes an order toward late.",
        "![SHAP importance](figures/shap_importance.png)",
        markdown_table(example_explanations(scores, rows, bundle)),
        "## Supplementary: retraining every month",
        "Not used to change the model. Each month is scored by the frozen model and by the same "
        "parameters retrained on everything known on that month's first day.",
        markdown_table(monthly_backtest(features, split, bundle)),
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "evaluation.md"
    report.write_text("\n\n".join(sections) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["test", "validation"], default="test")
    args = parser.parse_args()

    features = load_features()
    bundle = load_bundle()
    calibrators = joblib.load(CALIBRATORS_PATH)
    scores = score(features, args.split, bundle, calibrators)
    validation = score(features, "validation", bundle, calibrators)
    _, score_rows = period_for(args.split).masks(features)
    rows = features[score_rows]

    if args.split == "test":
        out_dir, predictions = REPORTS_DIR, ARTIFACTS_DIR / "test_predictions.csv"
    else:
        out_dir = ARTIFACTS_DIR / "dry_run"
        predictions = out_dir / "validation_predictions.csv"
    report = write_report(args.split, scores, validation, rows, features, bundle, out_dir)
    predictions.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(predictions, index=False)
    print(f"Wrote {report} and {predictions}")


if __name__ == "__main__":
    main()
