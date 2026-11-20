from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTICLES_PATH = PROJECT_ROOT / "data" / "articles.parquet"
LABELS_PATH = PROJECT_ROOT / "data" / "labels.parquet"
REFERENCE_PATH = PROJECT_ROOT / "data" / "reference_set.xlsx"
URLS_PATH = PROJECT_ROOT / "data" / "reference_urls.txt"
REFERENCE_COLUMNS = [
    "id",
    "url",
    "title",
    "date",
    "body",
    "cine",
    "tema_manual",
    "sentiment_manual",
    "perspectiva_manual",
]


def make_reference_set(
    articles_path: Path = ARTICLES_PATH,
    labels_path: Path = LABELS_PATH,
    reference_path: Path = REFERENCE_PATH,
    urls_path: Path = URLS_PATH,
) -> pd.DataFrame:
    articles = pd.read_parquet(articles_path)
    labels = pd.read_parquet(labels_path)
    if "url" not in articles.columns or "url" not in labels.columns:
        raise ValueError("Both input files must contain a 'url' column")

    eligible = articles.drop_duplicates(subset="url").merge(
        labels[["url"]].drop_duplicates(),
        on="url",
        how="inner",
    )
    if len(eligible) < 250:
        raise ValueError(
            f"Need 250 labeled articles; found {len(eligible)}"
        )

    selected = eligible.sample(n=250, random_state=7).reset_index(drop=True)
    body = selected["body"].fillna("").astype(str).str.slice(0, 2000)
    dates = pd.to_datetime(selected["date"], utc=True, errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    reference = pd.DataFrame(
        {
            "id": range(1, 251),
            "url": selected["url"],
            "title": selected["title"],
            "date": dates,
            "body": body,
            "cine": ["ambele"] * 40 + ["Alexia"] * 105 + ["colega"] * 105,
            "tema_manual": [""] * 250,
            "sentiment_manual": [""] * 250,
            "perspectiva_manual": [""] * 250,
        },
        columns=REFERENCE_COLUMNS,
    )

    reference_path.parent.mkdir(parents=True, exist_ok=True)
    urls_path.parent.mkdir(parents=True, exist_ok=True)
    reference.to_excel(reference_path, index=False, engine="openpyxl")
    urls_path.write_text("\n".join(reference["url"].astype(str)) + "\n", encoding="utf-8")
    return reference


if __name__ == "__main__":
    make_reference_set()
