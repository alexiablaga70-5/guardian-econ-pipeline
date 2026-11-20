import argparse
import os
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from dotenv import load_dotenv


GUARDIAN_API_URL = "https://content.guardianapis.com"
SECTIONS = ("business",)
START_DATE = date(2020, 1, 1)
MAX_ARTICLES = 5_000
COLUMNS = ["url", "title", "section", "date", "body"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "articles.parquet"


def fetch_articles(
    api_key: str,
    section: str,
    from_date: date,
    session: requests.Session | None = None,
    max_articles: int = MAX_ARTICLES,
    to_date: date | None = None,
    pages_fetched: list[int] | None = None,
    tag: str = "",
) -> list[dict]:
    http = session or requests.Session()
    articles: list[dict] = []
    page = 1

    while len(articles) < max_articles:
        endpoint = f"{GUARDIAN_API_URL}/search"
        params = {
            "api-key": api_key,
            "section": section,
            "from-date": from_date.isoformat(),
            "to-date": (to_date or date.today()).isoformat(),
            "show-fields": "bodyText",
            "page-size": 200,
            "page": page,
            "order-by": "oldest",
        }
        if tag:
            params["tag"] = tag

        try:
            response = http.get(
                endpoint,
                params=params,
                timeout=30,
            )
            response.raise_for_status()
        except requests.HTTPError as error:
            error_response = getattr(error, "response", None)
            response_url = getattr(error_response, "url", None) or endpoint
            status_code = getattr(error_response, "status_code", None)
            path = urlparse(response_url).path or "/search"
            if status_code is None:
                status_code = getattr(response, "status_code", "unknown")
            print(f"Guardian API HTTP {status_code}: {path}")
            break
        payload = response.json()["response"]
        if pages_fetched is not None:
            pages_fetched[0] += 1
        for item in payload["results"]:
            fields = item.get("fields") or {}
            articles.append(
                {
                    "url": item["webUrl"],
                    "title": item["webTitle"],
                    "section": item["sectionName"],
                    "date": item["webPublicationDate"],
                    "body": fields.get("bodyText", ""),
                }
            )
            if len(articles) >= max_articles:
                break

        if page >= payload["pages"]:
            break
        page += 1

    return articles


def load_existing(output_path: Path) -> pd.DataFrame:
    if not output_path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_parquet(output_path)


def month_starts(start: date, end: date):
    current = date(start.year, start.month, 1)
    while current <= end:
        yield current
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)


def month_end(month: date, end: date) -> date:
    if month.month == 12:
        next_month = date(month.year + 1, 1, 1)
    else:
        next_month = date(month.year, month.month + 1, 1)
    last_day = next_month.fromordinal(next_month.toordinal() - 1)
    return min(last_day, end)


def print_final_summary(data: pd.DataFrame) -> None:
    if data.empty:
        print("Final summary: 0 articles; date range: N/A")
        print("Sections: business=0, economics=0")
        return

    dates = pd.to_datetime(data["date"], utc=True, errors="coerce").dropna()
    if dates.empty:
        date_range = "N/A"
    else:
        date_range = f"{dates.min().date()} to {dates.max().date()}"

    sections = data["section"].fillna("").str.lower()
    business_count = sections.str.contains("business", regex=False).sum()
    economics_count = sections.str.contains("economics", regex=False).sum()
    print(f"Final summary: {len(data)} articles; date range: {date_range}")
    print(
        f"Sections: business={business_count}, economics={economics_count}"
    )


def collect_articles(
    output_path: Path = OUTPUT_PATH,
    session: requests.Session | None = None,
    max_articles: int = MAX_ARTICLES,
    per_month: int = 80,
    tag: str = "",
) -> pd.DataFrame:
    load_dotenv()
    api_key = os.getenv("GUARDIAN_API_KEY")
    if not api_key:
        raise RuntimeError("GUARDIAN_API_KEY is not set")

    if max_articles < 1:
        raise ValueError("max_articles must be at least 1")
    if per_month < 1:
        raise ValueError("per_month must be at least 1")

    combined = load_existing(output_path)
    combined["date"] = pd.to_datetime(combined["date"], utc=True, errors="coerce")
    pages_fetched = [0]
    today = date.today()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for month in month_starts(START_DATE, today):
        if len(combined) >= max_articles:
            break

        month_dates = pd.to_datetime(combined["date"], utc=True, errors="coerce")
        in_month = (
            (month_dates.dt.year == month.year)
            & (month_dates.dt.month == month.month)
        )
        month_count = int(in_month.sum())
        remaining_in_month = per_month - month_count
        if remaining_in_month <= 0:
            continue

        remaining_total = max_articles - len(combined)
        remaining = min(remaining_in_month, remaining_total)
        fetched: list[dict] = []
        quotas = [
            (remaining + len(SECTIONS) - index - 1) // len(SECTIONS)
            for index in range(len(SECTIONS))
        ]

        end = month_end(month, today)
        for section, quota in zip(SECTIONS, quotas):
            if quota <= 0:
                continue
            fetched.extend(
                fetch_articles(
                    api_key,
                    section,
                    month,
                    session,
                    quota,
                    end,
                    pages_fetched,
                    tag,
                )
            )

        if fetched:
            combined = pd.concat(
                [combined, pd.DataFrame(fetched, columns=COLUMNS)],
                ignore_index=True,
            )
        combined = combined.drop_duplicates(subset="url", keep="last")
        combined["date"] = pd.to_datetime(combined["date"], utc=True, errors="coerce")
        combined = combined.head(max_articles)
        combined.to_parquet(output_path, index=False)
        print(
            f"Pages fetched: {pages_fetched[0]}; "
            f"Articles saved so far: {len(combined)}"
        )

    combined.to_parquet(output_path, index=False)
    print_final_summary(combined)
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect articles from The Guardian")
    parser.add_argument("--max-articles", type=int, default=MAX_ARTICLES)
    parser.add_argument("--per-month", type=int, default=80)
    parser.add_argument("--tag", default="", help="Optional Guardian tag")
    args = parser.parse_args()
    if args.max_articles < 1:
        parser.error("--max-articles must be at least 1")
    if args.per_month < 1:
        parser.error("--per-month must be at least 1")
    collect_articles(
        max_articles=args.max_articles,
        per_month=args.per_month,
        tag=args.tag,
    )
