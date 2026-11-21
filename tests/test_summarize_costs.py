import csv
from pathlib import Path

import src.summarize_costs as summarize_costs


def test_main_generates_report_and_prints_totals(
    monkeypatch,
    tmp_path: Path,
    capsys,
):
    costs_path = tmp_path / "costs.csv"
    report_path = tmp_path / "docs" / "costs.md"
    with costs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "model", "input_tokens", "output_tokens"])
        writer.writerow(
            ["2026-01-01T00:00:00Z", "gemini-3.5-flash-lite", 100, 20]
        )
        writer.writerow(
            ["2026-01-01T00:01:00Z", "gemini-3.5-flash-lite", 40, 30]
        )
        writer.writerow(["2026-01-01T00:02:00Z", "other-model", 900, 900])

    monkeypatch.setattr(summarize_costs, "COSTS_PATH", costs_path)
    monkeypatch.setattr(summarize_costs, "REPORT_PATH", report_path)

    summarize_costs.main()

    report = report_path.read_text(encoding="utf-8")
    terminal_output = capsys.readouterr().out
    assert "| Gemini 3.5 Flash Lite (`gemini-3.5-flash-lite`) | 2 | 140 | 50 |" in report
    assert "0.0000390 USD" in report
    assert "0.00 USD" in report
    assert "The Guardian API" in report
    assert "data/llm_cache" in report
    assert "140" in terminal_output
    assert "50" in terminal_output
    assert "0.0000390 USD" in terminal_output
