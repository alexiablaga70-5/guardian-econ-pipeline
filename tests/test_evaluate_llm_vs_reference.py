from pathlib import Path

import pandas as pd

from src.evaluate_llm_vs_reference import evaluate_llm_vs_reference


def test_evaluate_metrics_and_write_excel_files(tmp_path: Path, capsys):
    reference_path = tmp_path / "reference_set_final.xlsx"
    labels_path = tmp_path / "labels.parquet"
    results_path = tmp_path / "llm_vs_reference.xlsx"
    errors_path = tmp_path / "llm_errors.xlsx"
    pd.DataFrame(
        [
            {
                "id": index + 1,
                "url": f"https://example.test/{index}",
                "title": f"Article {index}",
                "date": f"2026-01-0{index + 1}",
                "body": "Body " + ("x" * 1600),
                "tema_manual": manual_tema,
                "sentiment_manual": manual_sentiment,
                "perspectiva_manual": manual_perspective,
            }
            for index, (manual_tema, manual_sentiment, manual_perspective) in enumerate(
                [
                    ("inflatie", "negativ", "in_scadere"),
                    ("inflatie", "neutru", "stabila"),
                    ("energie", "pozitiv", "in_crestere"),
                    ("energie", "pozitiv", "stabila"),
                ]
            )
        ]
    ).to_excel(reference_path, index=False, engine="openpyxl")
    pd.DataFrame(
        [
            {
                "url": f"https://example.test/{index}",
                "tema": llm_tema,
                "sentiment": llm_sentiment,
                "perspectiva_economica": llm_perspective,
                "incredere": confidence,
            }
            for index, (
                llm_tema,
                llm_sentiment,
                llm_perspective,
                confidence,
            ) in enumerate(
                [
                    ("inflatie", "negativ", "in_scadere", 0.95),
                    ("energie", "negativ", "stabila", 0.9),
                    ("energie", "pozitiv", "in_crestere", 0.91),
                    ("energia", "pozitiv", "neclar", 0.5),
                ]
            )
        ]
    ).to_parquet(labels_path, index=False)

    results = evaluate_llm_vs_reference(
        reference_path=reference_path,
        labels_path=labels_path,
        results_path=results_path,
        errors_path=errors_path,
    )

    summary_workbook = pd.ExcelFile(results_path, engine="openpyxl")
    errors_workbook = pd.read_excel(errors_path, engine="openpyxl")
    assert set(summary_workbook.sheet_names) == {
        "summary",
        "tema",
        "sentiment",
        "perspectiva",
    }
    assert results["tema"]["metrics"]["articles_compared"] == 4
    assert results["tema"]["metrics"]["percent_agreement"] == 50
    assert results["tema"]["metrics"]["cohen_kappa"] == 0.2
    assert abs(results["tema"]["metrics"]["macro_f1"] - (7 / 18)) < 1e-12
    assert results["tema"]["confusion_matrix"].loc["inflatie", "energia"] == 0
    assert results["tema"]["metrics"]["above_0_9_articles"] == 2
    assert results["tema"]["metrics"]["at_or_below_0_9_articles"] == 2
    assert len(errors_workbook) == 2
    assert list(errors_workbook.columns) == [
        "id",
        "title",
        "body",
        "tema_manual",
        "tema",
        "incredere",
    ]
    assert errors_workbook["body"].str.len().max() == 1500
    assert "Summary:" in capsys.readouterr().out
