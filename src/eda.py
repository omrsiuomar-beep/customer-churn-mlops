import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CATEGORICAL, FIGURES, NUMERIC, ROOT
from src.data import clean, load_raw

sns.set_theme(style="whitegrid", palette="deep")
CHURN_COLORS = {"No": "#4c72b0", "Yes": "#c44e52"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=130)
    plt.close(fig)


def churn_rate(df, col):
    return df.groupby(col, observed=True)["Churn"].apply(lambda s: (s == "Yes").mean()).sort_values()


def missing_report(raw):
    blanks = raw["TotalCharges"].astype(str).str.strip().eq("").sum()
    nulls = raw.isna().sum()
    print(f"rows: {len(raw)}, columns: {raw.shape[1]}")
    print(f"real NaN values: {int(nulls.sum())}")
    print(f"blank TotalCharges (stored as text): {blanks}")
    return int(blanks)


def plot_target(df):
    counts = df["Churn"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(counts.index, counts.values, color=[CHURN_COLORS[k] for k in counts.index])
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 40, f"{value} ({value / len(df):.1%})", ha="center")
    ax.set_title("How many customers left?")
    ax.set_ylabel("Customers")
    save(fig, "01_churn_distribution.png")


def plot_numeric(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, NUMERIC):
        sns.histplot(df, x=col, hue="Churn", bins=30, palette=CHURN_COLORS, element="step", stat="density", common_norm=False, ax=ax)
        ax.set_title(col)
    save(fig, "02_numeric_histograms.png")


def plot_rates(df):
    cols = ["Contract", "InternetService", "PaymentMethod", "TechSupport", "OnlineSecurity", "SeniorCitizen"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    overall = (df["Churn"] == "Yes").mean()
    for ax, col in zip(axes.flat, cols):
        rates = churn_rate(df, col)
        ax.barh(rates.index, rates.values, color="#c44e52")
        ax.axvline(overall, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"Churn rate by {col}")
        ax.set_xlim(0, max(0.6, rates.max() + 0.05))
        for i, v in enumerate(rates.values):
            ax.text(v + 0.005, i, f"{v:.0%}", va="center", fontsize=9)
    save(fig, "03_churn_rate_by_category.png")


def plot_tenure_trend(df):
    bins = [0, 6, 12, 24, 36, 48, 60, 72]
    labels = ["0-6", "7-12", "13-24", "25-36", "37-48", "49-60", "61-72"]
    groups = pd.cut(df["tenure"], bins=bins, labels=labels, include_lowest=True)
    trend = df.assign(group=groups).groupby("group", observed=True)["Churn"].apply(lambda s: (s == "Yes").mean())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(trend.index.astype(str), trend.values, marker="o", color="#c44e52", linewidth=2)
    for x, y in zip(trend.index.astype(str), trend.values):
        ax.annotate(f"{y:.0%}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Churn rate")
    ax.set_title("Churn rate drops the longer a customer stays")
    save(fig, "04_churn_trend_by_tenure.png")
    return trend


def plot_correlation(df):
    encoded = pd.get_dummies(df[NUMERIC + ["Contract", "InternetService", "PaymentMethod"]], drop_first=False, dtype=float)
    encoded["Churn"] = (df["Churn"] == "Yes").astype(float)
    corr = encoded.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="RdBu_r", center=0, annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax)
    ax.set_title("Correlation matrix")
    save(fig, "05_correlation_heatmap.png")
    return corr["Churn"].drop("Churn").sort_values()


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    blanks = missing_report(raw)
    df = clean(raw)
    dupes = int(df.duplicated().sum())
    print(f"duplicate rows after dropping customerID: {dupes}")

    plot_target(df)
    plot_numeric(df)
    plot_rates(df)
    trend = plot_tenure_trend(df)
    corr = plot_correlation(df)

    summary = {
        "rows": len(raw),
        "blank_total_charges": blanks,
        "duplicates": dupes,
        "churn_rate": round(float((df["Churn"] == "Yes").mean()), 4),
        "churn_rate_by_contract": churn_rate(df, "Contract").round(4).to_dict(),
        "churn_rate_by_internet": churn_rate(df, "InternetService").round(4).to_dict(),
        "churn_rate_by_payment": churn_rate(df, "PaymentMethod").round(4).to_dict(),
        "churn_rate_by_tenure_group": {str(k): round(float(v), 4) for k, v in trend.items()},
        "median_by_churn": df.groupby("Churn")[NUMERIC].median().round(2).to_dict(),
        "correlation_with_churn": corr.round(3).to_dict(),
    }
    (ROOT / "reports" / "eda_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"churn rate: {summary['churn_rate']:.1%}")
    print("churn by contract:", {k: f"{v:.1%}" for k, v in summary["churn_rate_by_contract"].items()})
    print("strongest correlations with churn:")
    for name, value in list(corr.items())[:3] + list(corr.items())[-3:]:
        print(f"  {name:<40} {value:+.3f}")
    print(f"figures saved to {FIGURES.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
