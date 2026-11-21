import io
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
import requests

from src.collect_gdelt import collect_gdelt


def make_zip(rows: list[list[str]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "events.csv",
            "\n".join("\t".join(row) for row in rows),
        )
    return buffer.getvalue()


def make_row(event_id: str, source_url: str) -> list[str]:
    row = [""] * 61
    row[0] = event_id
    row[1] = "20200101"
    row[28] = "14"
    row[29] = "1"
    row[30] = "2.5"
    row[31] = "3"
    row[32] = "2"
    row[33] = "1"
    row[34] = "-0.5"
    row[53] = "US"
    row[59] = "20200101120000"
    row[60] = source_url
    return row


class MockResponse:
    status_code = 200

    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        pass


class MockSession:
    def __init__(self, content: bytes):
        self.content = content
        self.urls: list[str] = []

    def get(self, url: str, timeout: int):
        self.urls.append(url)
        return MockResponse(self.content)


def test_collect_gdelt_from_fake_zip(tmp_path: Path, capsys):
    events_dir = tmp_path / "events"
    processed_path = tmp_path / "processed.txt"
    session = MockSession(
        make_zip(
            [
                make_row("1001", "https://example.test/1"),
                make_row("1002", "https://example.test/2"),
                make_row("1001", "https://example.test/duplicate"),
            ]
        )
    )

    summary = collect_gdelt(
        events_dir=events_dir,
        processed_path=processed_path,
        max_files=1,
        session=session,
        end_time=datetime(2020, 1, 1, 12, tzinfo=timezone.utc),
    )

    dataset = ds.dataset(events_dir, format="parquet", partitioning="hive")
    saved = dataset.to_table().to_pandas()
    assert len(session.urls) == 1
    assert session.urls[0].endswith("20200101120000.export.CSV.zip")
    assert len(saved) == 2
    assert set(saved["GLOBALEVENTID"]) == {"1001", "1002"}
    assert summary == {
        "files_downloaded": 1,
        "files_skipped": 0,
        "total_rows": 2,
        "date_range": "2020-01-01 to 2020-01-01",
    }
    assert processed_path.read_text(encoding="utf-8").strip() == "20200101120000"
    assert "Final summary:" in capsys.readouterr().out


def test_collect_gdelt_skips_wrong_column_count(tmp_path: Path, capsys):
    row = make_row("2001", "https://example.test/bad")[:-1]
    session = MockSession(make_zip([row]))
    events_dir = tmp_path / "events"
    processed_path = tmp_path / "processed.txt"

    summary = collect_gdelt(
        events_dir=events_dir,
        processed_path=processed_path,
        max_files=1,
        session=session,
        end_time=datetime(2020, 1, 1, 12, tzinfo=timezone.utc),
    )

    assert summary["files_downloaded"] == 1
    assert summary["files_skipped"] == 1
    assert summary["total_rows"] == 0
    assert not list(events_dir.rglob("*.parquet"))
    assert "expected 61" in capsys.readouterr().out


def test_collect_gdelt_continues_after_network_error(tmp_path: Path):
    class FlakySession:
        def __init__(self):
            self.calls = 0

        def get(self, url: str, timeout: int):
            self.calls += 1
            if self.calls == 1:
                raise requests.ConnectionError("offline")
            return MockResponse(make_zip([make_row("3001", "https://example.test/3")]))

    session = FlakySession()
    summary = collect_gdelt(
        events_dir=tmp_path / "events",
        processed_path=tmp_path / "processed.txt",
        max_files=2,
        step_days=1,
        session=session,
        end_time=datetime(2020, 1, 2, 12, tzinfo=timezone.utc),
    )

    assert session.calls == 2
    assert summary["files_downloaded"] == 1
    assert summary["files_skipped"] == 1
    assert summary["total_rows"] == 1
