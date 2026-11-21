from pathlib import Path

import pandas as pd

from src.gdelt_vs_guardian import compare_sources


def test_compare_sources_fake_data_real_xlsx_and_png(tmp_path: Path, capsys):
    gdelt_dir = tmp_path / "gdelt" / "events"
    gdelt_dir.mkdir(parents=True)
    gdelt_path = gdelt_dir / "events.parquet"
    articles_path = tmp_path / "articles.parquet"
    labels_path = tmp_path / "labels.parquet"
    excel_path = tmp_path / "gdelt_vs_guardian.xlsx"
    chart_path = tmp_path / "gdelt_vs_guardian.png"

    gdelt = pd.DataFrame(
        [
            {
                "GLOBALEVENTID": "1",
                "DATEADDED": "20260101120000",
                "AvgTone": -2.0,
                "GoldsteinScale": 1.0,
            },
            {
                "GLOBALEVENTID": "2",
                "DATEADDED": "20260102120000",
                "AvgTone": -4.0,
                "GoldsteinScale": 3.0,
            },
            {
                "GLOBALEVENTID": "3",
                "DATEADDED": "20260201120000",
                "AvgTone": 2.0,
                "GoldsteinScale": 5.0,
            },
        ]
    )
    gdelt.to_parquet(gdelt_path, index=False)

    article_rows = [
        {
            "url": f"https://example.test/{index}",
            "date": pd.Timestamp("2026-01-15T12:00:00Z"),
        }
        for index in range(8)
    ]
    article_rows.extend(
        {
            "url": f"https://example.test/feb-{index}",
            "date": pd.Timestamp("2026-02-15T12:00:00Z"),
        }
        for index in range(2)
    )
    pd.DataFrame(article_rows).to_parquet(articles_path, index=False)

    label_rows = [
        {
            "url": article["url"],
            "sentiment": "negativ" if index % 2 == 0 else "pozitiv",
        }
        for index, article in enumerate(article_rows)
    ]
    pd.DataFrame(label_rows).to_parquet(labels_path, index=False)

    monthly, correlation, used_months = compare_sources(
        gdelt_events_path=gdelt_dir,
        articles_path=articles_path,
        labels_path=labels_path,
        excel_path=excel_path,
        chart_path=chart_path,
        min_articles=8,
    )

    assert len(monthly) == 2
    january = monthly.loc[monthly["month"] == "2026-01"].iloc[0]
    february = monthly.loc[monthly["month"] == "2026-02"].iloc[0]
    assert january["gdelt_event_count"] == 2
    assert january["gdelt_avg_tone"] == -3.0
    assert january["guardian_labeled_articles"] == 8
    assert january["guardian_negative_share"] == 50.0
    assert february["guardian_labeled_articles"] == 2
    assert bool(february["included_in_correlation"]) is False
    assert used_months == 1
    assert correlation is None
    assert excel_path.exists()
    assert chart_path.exists()
    workbook = pd.read_excel(excel_path, engine="openpyxl")
    assert len(workbook) == 2
    assert "Pearson correlation" in capsys.readouterr().out
