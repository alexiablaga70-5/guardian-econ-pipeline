from pathlib import Path

import pandas as pd

from src.make_reference_set import REFERENCE_COLUMNS, make_reference_set


def test_make_reference_set_uses_labeled_articles_and_expected_columns(
    tmp_path: Path,
):
    articles_path = tmp_path / "articles.parquet"
    labels_path = tmp_path / "labels.parquet"
    reference_path = tmp_path / "reference_set.xlsx"
    urls_path = tmp_path / "reference_urls.txt"
    article_rows = [
        {
            "url": f"https://example.test/{index}",
            "title": f"Article {index}",
            "date": pd.Timestamp("2026-01-01T12:00:00Z"),
            "body": f"{index} " + ("x" * 2100),
        }
        for index in range(260)
    ]
    pd.DataFrame(article_rows).to_parquet(articles_path, index=False)
    pd.DataFrame(
        [
            {"url": row["url"], "tema": "inflatie"}
            for row in article_rows[:255]
        ]
    ).to_parquet(labels_path, index=False)

    result = make_reference_set(
        articles_path=articles_path,
        labels_path=labels_path,
        reference_path=reference_path,
        urls_path=urls_path,
    )
    workbook = pd.read_excel(reference_path, engine="openpyxl")
    saved_urls = urls_path.read_text(encoding="utf-8").splitlines()

    assert len(result) == len(workbook) == len(saved_urls) == 250
    assert list(workbook.columns) == REFERENCE_COLUMNS
    assert workbook["date"].astype(str).eq("2026-01-01").all()
    assert set(result["url"]).issubset({row["url"] for row in article_rows[:255]})
    assert result["url"].tolist() == saved_urls
    assert set(result["body"].str.len()) == {2000}
    assert result["cine"].iloc[:40].eq("ambele").all()
    assert result["cine"].iloc[40:145].eq("Alexia").all()
    assert result["cine"].iloc[145:].eq("colega").all()
    assert result[["tema_manual", "sentiment_manual", "perspectiva_manual"]].eq("").all().all()
