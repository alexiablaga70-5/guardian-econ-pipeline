import argparse
import csv
import io
import re
import uuid
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GDELT_DIR = PROJECT_ROOT / "data" / "gdelt"
EVENTS_DIR = GDELT_DIR / "events"
PROCESSED_PATH = GDELT_DIR / "processed.txt"
URL_TEMPLATE = "http://data.gdeltproject.org/gdeltv2/{timestamp}.export.CSV.zip"
START_TIME = datetime(2020, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
COLUMN_INDEXES = [0, 1, 28, 29, 30, 31, 32, 33, 34, 53, 59, 60]
COLUMN_NAMES = [
    "GLOBALEVENTID",
    "SQLDATE",
    "EventRootCode",
    "QuadClass",
    "GoldsteinScale",
    "NumMentions",
    "NumSources",
    "NumArticles",
    "AvgTone",
    "ActionGeo_CountryCode",
    "DATEADDED",
    "SOURCEURL",
]
CHUNK_SIZE = 50_000


def candidate_files(step_days: int, end_time: datetime) -> list[str]:
    candidates = []
    current = START_TIME
    while current <= end_time:
        candidates.append(current.strftime("%Y%m%d%H%M%S"))
        current += timedelta(days=step_days)
    return candidates


def read_processed(processed_path: Path) -> set[str]:
    if not processed_path.exists():
        return set()
    return {
        line.strip()
        for line in processed_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def mark_processed(processed_path: Path, timestamp: str) -> None:
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    with processed_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp}\n")


def load_existing_event_ids(events_dir: Path) -> set[str]:
    parquet_files = list(events_dir.rglob("*.parquet")) if events_dir.exists() else []
    if not parquet_files:
        return set()

    dataset = ds.dataset(events_dir, format="parquet", partitioning="hive")
    event_ids: set[str] = set()
    for batch in dataset.scanner(
        columns=["GLOBALEVENTID"],
        batch_size=CHUNK_SIZE,
    ).to_batches():
        event_ids.update(str(value) for value in batch.column(0).to_pylist() if value is not None)
    return event_ids


def dataset_stats(events_dir: Path) -> tuple[int, str | None, str | None]:
    parquet_files = list(events_dir.rglob("*.parquet")) if events_dir.exists() else []
    if not parquet_files:
        return 0, None, None

    dataset = ds.dataset(events_dir, format="parquet", partitioning="hive")
    total_rows = dataset.count_rows()
    minimum: str | None = None
    maximum: str | None = None
    for batch in dataset.scanner(
        columns=["SQLDATE"],
        batch_size=CHUNK_SIZE,
    ).to_batches():
        dates = pd.to_datetime(
            pd.Series(batch.column(0).to_pylist(), dtype="string"),
            format="%Y%m%d",
            errors="coerce",
        ).dropna()
        if dates.empty:
            continue
        batch_min = dates.min().strftime("%Y-%m-%d")
        batch_max = dates.max().strftime("%Y-%m-%d")
        minimum = min(minimum, batch_min) if minimum else batch_min
        maximum = max(maximum, batch_max) if maximum else batch_max
    return total_rows, minimum, maximum


def validate_archive(content: bytes) -> tuple[zipfile.ZipFile, str]:
    archive = zipfile.ZipFile(io.BytesIO(content))
    members = [name for name in archive.namelist() if not name.endswith("/")]
    if len(members) != 1:
        archive.close()
        raise ValueError("archive must contain exactly one CSV file")

    try:
        row_count = 0
        with archive.open(members[0]) as csv_file:
            text_file = io.TextIOWrapper(csv_file, encoding="utf-8")
            for row_count, row in enumerate(
                csv.reader(text_file, delimiter="\t"),
                start=1,
            ):
                if len(row) != 61:
                    raise ValueError(
                        f"CSV row {row_count} has {len(row)} columns; expected 61"
                    )
    except (UnicodeDecodeError, csv.Error) as error:
        archive.close()
        raise ValueError("CSV file is not valid UTF-8 tab-separated data") from error
    except ValueError:
        archive.close()
        raise
    if row_count == 0:
        archive.close()
        raise ValueError("CSV file is empty")
    return archive, members[0]


def write_chunk(
    chunk: pd.DataFrame,
    events_dir: Path,
    seen_ids: set[str],
    basename: str,
    chunk_number: int,
) -> tuple[int, list[str], list[str]]:
    selected = chunk.iloc[:, COLUMN_INDEXES].copy()
    selected.columns = COLUMN_NAMES
    selected["GLOBALEVENTID"] = selected["GLOBALEVENTID"].astype("string")
    selected = selected.drop_duplicates(subset="GLOBALEVENTID", keep="first")
    selected = selected.loc[
        ~selected["GLOBALEVENTID"].isin(seen_ids)
    ].copy()
    if selected.empty:
        return 0, [], []

    dates = pd.to_datetime(
        selected["SQLDATE"].astype("string"),
        format="%Y%m%d",
        errors="coerce",
    )
    date_values = dates.dropna()
    date_min = date_values.min().strftime("%Y-%m-%d") if not date_values.empty else None
    date_max = date_values.max().strftime("%Y-%m-%d") if not date_values.empty else None
    selected["year"] = dates.dt.strftime("%Y").fillna("unknown")
    selected["month"] = dates.dt.strftime("%m").fillna("unknown")

    table = pa.Table.from_pandas(selected, preserve_index=False)
    ds.write_dataset(
        table,
        base_dir=events_dir,
        format="parquet",
        partitioning=["year", "month"],
        partitioning_flavor="hive",
        basename_template=f"{basename}-{chunk_number}-{{i}}.parquet",
        existing_data_behavior="overwrite_or_ignore",
    )
    seen_ids.update(str(value) for value in selected["GLOBALEVENTID"])
    return len(selected), [date_min] if date_min else [], [date_max] if date_max else []


def collect_gdelt(
    events_dir: Path = EVENTS_DIR,
    processed_path: Path = PROCESSED_PATH,
    max_files: int | None = None,
    step_days: int = 7,
    session: Any | None = None,
    end_time: datetime | None = None,
) -> dict[str, Any]:
    if step_days < 1:
        raise ValueError("step_days must be at least 1")
    if max_files is not None and max_files < 1:
        raise ValueError("max_files must be at least 1")

    http = session or requests.Session()
    processed = read_processed(processed_path)
    candidates = candidate_files(
        step_days,
        end_time or datetime.now(timezone.utc),
    )
    already_processed = [timestamp for timestamp in candidates if timestamp in processed]
    pending = [timestamp for timestamp in candidates if timestamp not in processed]
    if max_files is not None:
        pending = pending[:max_files]

    events_dir.mkdir(parents=True, exist_ok=True)
    seen_ids = load_existing_event_ids(events_dir)
    total_rows, minimum_date, maximum_date = dataset_stats(events_dir)
    files_downloaded = 0
    files_skipped = len(already_processed)
    if already_processed:
        print(f"Progress: {len(already_processed)} previously processed files skipped")

    for index, timestamp in enumerate(pending, start=1):
        url = URL_TEMPLATE.format(timestamp=timestamp)
        try:
            response = http.get(url, timeout=60)
            response.raise_for_status()
        except requests.HTTPError as error:
            status = error.response.status_code if error.response is not None else "unknown"
            print(f"Warning: skipped {timestamp}; HTTP {status}")
            files_skipped += 1
            if status == 404:
                mark_processed(processed_path, timestamp)
                processed.add(timestamp)
            print(
                f"Progress: {index}/{len(pending)} new files; "
                f"downloaded={files_downloaded}, skipped={files_skipped}, rows={total_rows}"
            )
            continue
        except requests.RequestException as error:
            print(f"Warning: skipped {timestamp}; network error ({type(error).__name__})")
            files_skipped += 1
            print(
                f"Progress: {index}/{len(pending)} new files; "
                f"downloaded={files_downloaded}, skipped={files_skipped}, rows={total_rows}"
            )
            continue

        files_downloaded += 1
        basename = f"{timestamp}-{uuid.uuid4().hex}"
        try:
            archive, member = validate_archive(response.content)
        except (ValueError, zipfile.BadZipFile) as error:
            print(f"Warning: skipped {timestamp}; invalid archive: {error}")
            files_skipped += 1
            mark_processed(processed_path, timestamp)
            processed.add(timestamp)
            print(
                f"Progress: {index}/{len(pending)} new files; "
                f"downloaded={files_downloaded}, skipped={files_skipped}, rows={total_rows}"
            )
            continue

        new_rows = 0
        file_min: str | None = None
        file_max: str | None = None
        try:
            with archive, archive.open(member) as csv_file:
                reader = pd.read_csv(
                    csv_file,
                    sep="\t",
                    header=None,
                    dtype="string",
                    chunksize=CHUNK_SIZE,
                )
                for chunk_number, chunk in enumerate(reader):
                    if chunk.shape[1] != 61:
                        raise pd.errors.ParserError(
                            f"CSV chunk has {chunk.shape[1]} columns; expected 61"
                        )
                    rows, minima, maxima = write_chunk(
                        chunk,
                        events_dir,
                        seen_ids,
                        basename,
                        chunk_number,
                    )
                    new_rows += rows
                    if minima:
                        file_min = min(file_min, minima[0]) if file_min else minima[0]
                    if maxima:
                        file_max = max(file_max, maxima[0]) if file_max else maxima[0]
        except (pd.errors.ParserError, UnicodeDecodeError) as error:
            for partial_file in events_dir.rglob(f"{basename}-*.parquet"):
                partial_file.unlink()
            print(f"Warning: skipped {timestamp}; invalid CSV: {error}")
            files_skipped += 1
            mark_processed(processed_path, timestamp)
            processed.add(timestamp)
            print(
                f"Progress: {index}/{len(pending)} new files; "
                f"downloaded={files_downloaded}, skipped={files_skipped}, rows={total_rows}"
            )
            continue

        total_rows += new_rows
        if file_min:
            minimum_date = min(minimum_date, file_min) if minimum_date else file_min
        if file_max:
            maximum_date = max(maximum_date, file_max) if maximum_date else file_max
        mark_processed(processed_path, timestamp)
        processed.add(timestamp)
        print(
            f"Progress: {index}/{len(pending)} new files; "
            f"downloaded={files_downloaded}, skipped={files_skipped}, rows={total_rows}"
        )

    date_range = (
        f"{minimum_date} to {maximum_date}"
        if minimum_date and maximum_date
        else "N/A"
    )
    summary = {
        "files_downloaded": files_downloaded,
        "files_skipped": files_skipped,
        "total_rows": total_rows,
        "date_range": date_range,
    }
    print(
        "Final summary: "
        f"files downloaded={files_downloaded}, files skipped={files_skipped}, "
        f"total rows={total_rows}, date range={date_range}"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect sampled GDELT 2.0 event files")
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--step-days", type=int, default=7)
    args = parser.parse_args()
    if args.max_files is not None and args.max_files < 1:
        parser.error("--max-files must be at least 1")
    if args.step_days < 1:
        parser.error("--step-days must be at least 1")
    collect_gdelt(max_files=args.max_files, step_days=args.step_days)
