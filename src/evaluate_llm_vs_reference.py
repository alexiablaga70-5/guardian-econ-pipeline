from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = PROJECT_ROOT / "data" / "reference_set_final.xlsx"
LABELS_PATH = PROJECT_ROOT / "data" / "labels.parquet"
RESULTS_PATH = PROJECT_ROOT / "data" / "llm_vs_reference.xlsx"
ERRORS_PATH = PROJECT_ROOT / "data" / "llm_errors.xlsx"
FIELD_PAIRS = {
    "tema": ("tema_manual", "tema"),
    "sentiment": ("sentiment_manual", "sentiment"),
    "perspectiva": ("perspectiva_manual", "perspectiva_economica"),
}


def _class_metrics(actual: list[Any], predicted: list[Any]) -> pd.DataFrame:
    classes = sorted(set(actual) | set(predicted), key=str)
    rows = []
    for label in classes:
        true_positive = sum(
            observed == label and estimate == label
            for observed, estimate in zip(actual, predicted)
        )
        actual_count = sum(observed == label for observed in actual)
        predicted_count = sum(estimate == label for estimate in predicted)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / actual_count if actual_count else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        rows.append(
            {
                "class": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": actual_count,
            }
        )
    return pd.DataFrame(rows, columns=["class", "precision", "recall", "f1", "support"])


def _cohen_kappa(actual: list[Any], predicted: list[Any]) -> float | None:
    total = len(actual)
    if not total:
        return None

    classes = set(actual) | set(predicted)
    observed_agreement = sum(
        observed == estimate
        for observed, estimate in zip(actual, predicted)
    ) / total
    expected_agreement = sum(
        sum(value == label for value in actual)
        * sum(value == label for value in predicted)
        for label in classes
    ) / (total * total)
    if expected_agreement == 1:
        return 1.0 if observed_agreement == 1 else 0.0
    return (observed_agreement - expected_agreement) / (1 - expected_agreement)


def _agreement_percent(actual: list[Any], predicted: list[Any]) -> float | None:
    if not actual:
        return None
    return 100 * sum(
        observed == estimate
        for observed, estimate in zip(actual, predicted)
    ) / len(actual)


def _evaluate_field(
    joined: pd.DataFrame,
    manual_column: str,
    llm_column: str,
) -> dict[str, Any]:
    compared = joined.loc[
        joined[manual_column].notna()
        & joined[llm_column].notna()
        & joined[manual_column].astype(str).str.strip().ne("")
        & joined[llm_column].astype(str).str.strip().ne("")
    ].copy()
    actual = compared[manual_column].tolist()
    predicted = compared[llm_column].tolist()
    class_metrics = _class_metrics(actual, predicted)
    macro_f1 = (
        float(class_metrics["f1"].mean())
        if not class_metrics.empty
        else None
    )
    metrics: dict[str, Any] = {
        "articles_compared": len(compared),
        "percent_agreement": _agreement_percent(actual, predicted),
        "cohen_kappa": _cohen_kappa(actual, predicted),
        "macro_f1": macro_f1,
    }

    confidence = pd.to_numeric(compared["incredere"], errors="coerce")
    agreements = pd.Series(
        [left == right for left, right in zip(actual, predicted)],
        index=compared.index,
    )
    for threshold_label, mask in (
        ("above_0_9", confidence > 0.9),
        ("at_or_below_0_9", confidence <= 0.9),
    ):
        bucket_agreements = agreements[mask.fillna(False)]
        metrics[f"{threshold_label}_articles"] = int(len(bucket_agreements))
        metrics[f"{threshold_label}_agreement_percent"] = (
            100 * float(bucket_agreements.mean())
            if not bucket_agreements.empty
            else None
        )

    return {
        "metrics": metrics,
        "per_class": class_metrics,
        "compared": compared,
    }


def _excel_safe(frame: pd.DataFrame) -> pd.DataFrame:
    safe = frame.copy()
    for column in safe.select_dtypes(include=["datetimetz"]).columns:
        safe[column] = safe[column].dt.tz_localize(None)
    return safe


def _print_field(name: str, result: dict[str, Any]) -> None:
    print(f"\n{name}")
    print(pd.DataFrame([result["metrics"]]).to_string(index=False))
    print("Per-class metrics:")
    print(result["per_class"].to_string(index=False))


def evaluate_llm_vs_reference(
    reference_path: Path = REFERENCE_PATH,
    labels_path: Path = LABELS_PATH,
    results_path: Path = RESULTS_PATH,
    errors_path: Path = ERRORS_PATH,
) -> dict[str, Any]:
    reference = pd.read_excel(reference_path, engine="openpyxl")
    labels = pd.read_parquet(labels_path)
    required_reference = {"url", *(manual for manual, _ in FIELD_PAIRS.values())}
    required_labels = {
        "url",
        "incredere",
        *(llm for _, llm in FIELD_PAIRS.values()),
    }
    if not required_reference.issubset(reference.columns):
        raise ValueError("Reference workbook is missing required comparison columns")
    if not required_labels.issubset(labels.columns):
        raise ValueError("Labels parquet is missing required comparison columns")

    joined = reference.merge(
        labels[list(required_labels)].drop_duplicates(subset="url", keep="last"),
        on="url",
        how="inner",
    )
    results = {
        field: _evaluate_field(joined, manual, llm)
        for field, (manual, llm) in FIELD_PAIRS.items()
    }

    tema_compared = results["tema"]["compared"]
    tema_classes = sorted(
        set(tema_compared["tema_manual"].tolist())
        | set(tema_compared["tema"].tolist()),
        key=str,
    )
    tema_confusion = pd.crosstab(
        tema_compared["tema_manual"],
        tema_compared["tema"],
        rownames=["manual"],
        colnames=["LLM"],
    ).reindex(index=tema_classes, columns=tema_classes, fill_value=0)
    results["tema"]["confusion_matrix"] = tema_confusion

    summary = pd.DataFrame(
        [
            {"field": field, **result["metrics"]}
            for field, result in results.items()
        ]
    )
    print(f"Articles joined: {len(joined)}")
    print("Summary:")
    print(summary.to_string(index=False))
    for field, result in results.items():
        _print_field(field, result)
    print("\nTema confusion matrix (rows=manual, columns=LLM):")
    print(tema_confusion.to_string())

    errors = joined.loc[
        joined["tema_manual"].notna()
        & joined["tema"].notna()
        & joined["tema_manual"].astype(str).str.strip().ne("")
        & joined["tema"].astype(str).str.strip().ne("")
        & joined["tema_manual"].ne(joined["tema"]),
        ["id", "title", "body", "tema_manual", "tema", "incredere"],
    ].copy()
    errors["body"] = errors["body"].fillna("").astype(str).str.slice(0, 1500)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(results_path, engine="openpyxl") as writer:
        _excel_safe(summary).to_excel(
            writer,
            sheet_name="summary",
            index=False,
        )
        for field, result in results.items():
            metrics_frame = pd.DataFrame(
                [
                    {"metric": name, "value": value}
                    for name, value in result["metrics"].items()
                ]
            )
            metrics_frame.to_excel(
                writer,
                sheet_name=field,
                index=False,
                startrow=0,
            )
            class_start = len(metrics_frame) + 2
            result["per_class"].to_excel(
                writer,
                sheet_name=field,
                index=False,
                startrow=class_start,
            )
            if field == "tema":
                matrix_start = class_start + len(result["per_class"]) + 2
                tema_confusion.to_excel(
                    writer,
                    sheet_name=field,
                    startrow=matrix_start,
                )
    _excel_safe(errors).to_excel(errors_path, index=False, engine="openpyxl")
    results["summary"] = summary
    results["errors"] = errors
    return results


if __name__ == "__main__":
    evaluate_llm_vs_reference()
