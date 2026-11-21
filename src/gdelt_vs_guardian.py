import argparse
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GDELT_EVENTS_PATH = PROJECT_ROOT / "data" / "gdelt" / "events"
ARTICLES_PATH = PROJECT_ROOT / "data" / "articles.parquet"
LABELS_PATH = PROJECT_ROOT / "data" / "labels.parquet"
EXCEL_PATH = PROJECT_ROOT / "data" / "gdelt_vs_guardian.xlsx"
CHART_PATH = PROJECT_ROOT / "data" / "gdelt_vs_guardian.png"


def pearson_correlation(left: pd.Series, right: pd.Series) -> float | None:
    compared = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(compared) < 2:
        return None

    left_values = compared["left"].astype(float)
    right_values = compared["right"].astype(float)
    left_mean = left_values.mean()
    right_mean = right_values.mean()
    left_delta = left_values - left_mean
    right_delta = right_values - right_mean
    denominator = (
        (left_delta.pow(2).sum() * right_delta.pow(2).sum())
        ** 0.5
    )
    if denominator == 0:
        return None
    return float((left_delta * right_delta).sum() / denominator)


def timezone_free_excel(frame: pd.DataFrame) -> pd.DataFrame:
    safe = frame.copy()
    for column in safe.columns:
        dtype = safe[column].dtype
        if isinstance(dtype, pd.DatetimeTZDtype):
            safe[column] = safe[column].dt.tz_localize(None)
    return safe


def read_gdelt(events_path: Path) -> pd.DataFrame:
    if not events_path.exists():
        return pd.DataFrame(
            columns=["DATEADDED", "AvgTone", "GoldsteinScale"]
        )
    return pd.read_parquet(events_path)


def build_monthly_table(
    gdelt_events_path: Path,
    articles_path: Path,
    labels_path: Path,
    min_articles: int,
) -> tuple[pd.DataFrame, float | None, int]:
    gdelt = read_gdelt(gdelt_events_path)
    if not gdelt.empty:
        gdelt = gdelt.copy()
        gdelt["month"] = (
            gdelt["DATEADDED"].astype("string").str[:6]
        )
        gdelt["month"] = gdelt["month"].str.replace(
            r"(\d{4})(\d{2})",
            r"\1-\2",
            regex=True,
        )
        gdelt["AvgTone"] = pd.to_numeric(gdelt["AvgTone"], errors="coerce")
        gdelt["GoldsteinScale"] = pd.to_numeric(
            gdelt["GoldsteinScale"],
            errors="coerce",
        )
        gdelt_monthly = (
            gdelt.groupby("month", dropna=True)
            .agg(
                gdelt_avg_tone=("AvgTone", "mean"),
                gdelt_avg_goldstein=("GoldsteinScale", "mean"),
                gdelt_event_count=("GLOBALEVENTID", "count"),
            )
            .reset_index()
        )
    else:
        gdelt_monthly = pd.DataFrame(
            columns=[
                "month",
                "gdelt_avg_tone",
                "gdelt_avg_goldstein",
                "gdelt_event_count",
            ]
        )

    articles = pd.read_parquet(articles_path)
    labels = pd.read_parquet(labels_path)
    guardian = labels.merge(
        articles[["url", "date"]],
        on="url",
        how="inner",
    )
    guardian["month"] = (
        pd.to_datetime(guardian["date"], utc=True, errors="coerce")
        .dt.strftime("%Y-%m")
        .fillna("")
    )
    guardian["is_negative"] = (
        guardian["sentiment"].astype("string").str.lower() == "negativ"
    )
    guardian_monthly = (
        guardian[guardian["month"] != ""]
        .groupby("month", dropna=False)
        .agg(
            guardian_negative_share=("is_negative", "mean"),
            guardian_labeled_articles=("url", "count"),
        )
        .reset_index()
    )
    guardian_monthly["guardian_negative_share"] = (
        guardian_monthly["guardian_negative_share"] * 100
    )

    monthly = gdelt_monthly.merge(
        guardian_monthly,
        on="month",
        how="outer",
    ).sort_values("month", na_position="last")
    for column in (
        "gdelt_event_count",
        "guardian_labeled_articles",
    ):
        monthly[column] = monthly[column].fillna(0).astype(int)
    monthly["included_in_correlation"] = (
        monthly["guardian_labeled_articles"].fillna(0) >= min_articles
    )
    eligible = monthly[
        monthly["included_in_correlation"]
        & monthly["gdelt_avg_tone"].notna()
        & monthly["guardian_negative_share"].notna()
    ]
    correlation = pearson_correlation(
        eligible["gdelt_avg_tone"],
        eligible["guardian_negative_share"],
    )
    monthly["month"] = pd.to_datetime(
        monthly["month"],
        format="%Y-%m",
        errors="coerce",
    ).dt.to_period("M").astype("string")
    monthly = monthly[
        [
            "month",
            "gdelt_avg_tone",
            "gdelt_avg_goldstein",
            "gdelt_event_count",
            "guardian_negative_share",
            "guardian_labeled_articles",
            "included_in_correlation",
        ]
    ]
    return monthly, correlation, len(eligible)


def save_chart(monthly: pd.DataFrame, chart_path: Path) -> None:
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    chart = monthly.dropna(subset=["month"]).copy()
    chart["plot_month"] = pd.to_datetime(
        chart["month"].astype("string"),
        format="%Y-%m",
        errors="coerce",
    )

    figure, first_axis = plt.subplots(figsize=(12, 6))
    first_axis.plot(
        chart["plot_month"],
        chart["gdelt_avg_tone"],
        color="tab:blue",
        marker="o",
        label="GDELT AvgTone",
    )
    first_axis.set_xlabel("Month")
    first_axis.set_ylabel("GDELT AvgTone", color="tab:blue")
    first_axis.tick_params(axis="y", labelcolor="tab:blue")

    second_axis = first_axis.twinx()
    second_axis.plot(
        chart["plot_month"],
        chart["guardian_negative_share"],
        color="tab:red",
        marker="o",
        label="Guardian negative share (%)",
    )
    second_axis.set_ylabel(
        "Guardian negative share (%)",
        color="tab:red",
    )
    second_axis.tick_params(axis="y", labelcolor="tab:red")

    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(chart_path, dpi=150)
    plt.close(figure)


def compare_sources(
    gdelt_events_path: Path = GDELT_EVENTS_PATH,
    articles_path: Path = ARTICLES_PATH,
    labels_path: Path = LABELS_PATH,
    excel_path: Path = EXCEL_PATH,
    chart_path: Path = CHART_PATH,
    min_articles: int = 8,
) -> tuple[pd.DataFrame, float | None, int]:
    if min_articles < 1:
        raise ValueError("min_articles must be at least 1")

    monthly, correlation, used_months = build_monthly_table(
        gdelt_events_path,
        articles_path,
        labels_path,
        min_articles,
    )
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    timezone_free_excel(monthly).to_excel(
        excel_path,
        index=False,
        engine="openpyxl",
    )
    save_chart(monthly, chart_path)

    correlation_text = (
        f"{correlation:.6f}" if correlation is not None else "N/A"
    )
    print(
        "Pearson correlation GDELT AvgTone vs Guardian negative share: "
        f"{correlation_text}; months used: {used_months}"
    )
    return monthly, correlation, used_months


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare monthly GDELT events with Guardian labels"
    )
    parser.add_argument("--min-articles", type=int, default=8)
    args = parser.parse_args()
    if args.min_articles < 1:
        parser.error("--min-articles must be at least 1")
    compare_sources(min_articles=args.min_articles)
