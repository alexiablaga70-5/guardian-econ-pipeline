import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


GUARDIAN_API_URL = "https://content.guardianapis.com"
SECTIONS = ("business", "economics")
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
) -> list[dict]:
    http = session or requests.Session()
    articles: list[dict] = []
    page = 1

    while len(articles) < MAX_ARTICLES:
        response = http.get(
            f"{GUARDIAN_API_URL}/{section}",
            params={
                "api-key": api_key,
                "from-date": from_date.isoformat(),
                "to-date": date.today().isoformat(),
                "show-fields": "bodyText",
                "page-size": 200,
                "page": page,
                "order-by": "oldest",
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()["response"]
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
            if len(articles) >= MAX_ARTICLES:
                break

        if page >= payload["pages"]:
            break
        page += 1

    return articles


def load_existing(output_path: Path) -> pd.DataFrame:
    if not output_path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_parquet(output_path)


def collect_articles(
    output_path: Path = OUTPUT_PATH,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    load_dotenv()
    api_key = os.getenv("GUARDIAN_API_KEY")
    if not api_key:
        raise RuntimeError("GUARDIAN_API_KEY is not set")

    existing = load_existing(output_path)
    if existing.empty:
        from_date = START_DATE
    else:
        existing_dates = pd.to_datetime(existing["date"], utc=True, errors="coerce")
        latest = existing_dates.max()
        if pd.isna(latest):
            from_date = START_DATE
        else:
            from_date = max(START_DATE, (latest.date() - timedelta(days=1)))

    fetched: list[dict] = []
    for section in SECTIONS:
        remaining = MAX_ARTICLES - len(existing)
        if remaining <= 0:
            break
        fetched.extend(fetch_articles(api_key, section, from_date, session)[:remaining])

    combined = pd.concat(
        [existing, pd.DataFrame(fetched, columns=COLUMNS)],
        ignore_index=True,
    )
    combined = combined.drop_duplicates(subset="url", keep="last")
    combined = combined.head(MAX_ARTICLES)
    combined["date"] = pd.to_datetime(combined["date"], utc=True, errors="coerce")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    return combined


if __name__ == "__main__":
    collect_articles()
