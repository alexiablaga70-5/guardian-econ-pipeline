import argparse
import csv
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from pydantic import ValidationError

from .schema import ArticleLabel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTICLES_PATH = PROJECT_ROOT / "data" / "articles.parquet"
LABELS_PATH = PROJECT_ROOT / "data" / "labels.parquet"
CACHE_DIR = PROJECT_ROOT / "data" / "llm_cache"
INVALID_PATH = PROJECT_ROOT / "data" / "invalid.jsonl"
FAILED_PATH = PROJECT_ROOT / "data" / "failed.jsonl"
COSTS_PATH = PROJECT_ROOT / "data" / "costs.csv"
SEED = 42
RETRY_DELAYS = (10, 30, 60, 120, 240)


def build_prompt(article: pd.Series) -> str:
    body = str(article.get("body", ""))[:4000]
    title = str(article.get("title", ""))
    return (
        "Clasifica articolul in JSON valid conform schemei. Tema trebuie sa "
        "reflecte subiectul principal. Daca articolul nu este in principal "
        "despre economie, afaceri sau finante (de exemplu muzica, mancare, "
        "calatorii, sport sau cultura), seteaza tema la \"altele\" si "
        "perspectiva_economica la \"neclar\".\n"
        f"Titlu: {title}\n"
        f"Text:\n{body}"
    )


def cache_key(prompt: str, model: str) -> str:
    return hashlib.sha256(f"{prompt}\0{model}".encode("utf-8")).hexdigest()


def read_cache(prompt: str, model: str) -> dict[str, Any] | None:
    path = CACHE_DIR / f"{cache_key(prompt, model)}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def write_cache(prompt: str, model: str, response: dict[str, Any]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{cache_key(prompt, model)}.json"
    path.write_text(json.dumps(response, ensure_ascii=False), encoding="utf-8")


def sample_across_months(
    articles: pd.DataFrame,
    limit: int,
    seed: int = SEED,
) -> pd.DataFrame:
    if articles.empty or limit <= 0:
        return articles.iloc[0:0].copy()

    dated = articles.copy()
    dated["_month"] = pd.to_datetime(
        dated["date"], utc=True, errors="coerce"
    ).dt.strftime("%Y-%m")
    dated = dated[dated["_month"].notna()]

    month_groups = [
        group.drop(columns=["_month"]).sample(
            frac=1,
            random_state=seed + index,
        )
        for index, (_, group) in enumerate(dated.groupby("_month", sort=True))
    ]
    if not month_groups:
        return dated.sample(n=min(limit, len(dated)), random_state=seed).copy()

    remaining = min(limit, len(dated))
    month_count = len(month_groups)
    selected_indexes: list[Any] = []
    positions = [0] * month_count

    for selection_index in range(remaining):
        target_month = min(
            month_count - 1,
            int((selection_index + 0.5) * month_count / remaining),
        )
        for offset in range(month_count):
            month_index = (target_month + offset) % month_count
            month_group = month_groups[month_index]
            if positions[month_index] < len(month_group):
                selected_indexes.append(month_group.index[positions[month_index]])
                positions[month_index] += 1
                break

    result = dated.loc[selected_indexes].drop(columns=["_month"]).copy()
    result = result.sample(frac=1, random_state=seed)
    return result


def load_labels(labels_path: Path) -> pd.DataFrame:
    if not labels_path.exists():
        return pd.DataFrame(columns=["url"])
    return pd.read_parquet(labels_path)


def append_labels(labels: list[dict[str, Any]], labels_path: Path) -> pd.DataFrame:
    existing = load_labels(labels_path)
    new = pd.DataFrame(labels)
    combined = pd.concat([existing, new], ignore_index=True)
    if "url" in combined.columns:
        combined = combined.drop_duplicates(subset="url", keep="last")
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(labels_path, index=False)
    return combined


def append_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> None:
    COSTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not COSTS_PATH.exists()
    with COSTS_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(
                ["timestamp", "model", "input_tokens", "output_tokens"]
            )
        writer.writerow(
            [
                datetime.now(timezone.utc).isoformat(),
                model,
                input_tokens,
                output_tokens,
            ]
        )


def append_invalid(article: pd.Series, answer: str, error: str) -> None:
    INVALID_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "url": str(article["url"]),
        "answer": answer,
        "error": error,
    }
    with INVALID_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def status_code(error: Exception) -> int | None:
    candidates = [
        getattr(error, "status_code", None),
        getattr(error, "code", None),
    ]
    response = getattr(error, "response", None)
    if response is not None:
        candidates.append(getattr(response, "status_code", None))
    for candidate in candidates:
        if isinstance(candidate, int):
            return candidate
        if isinstance(candidate, str) and candidate.isdigit():
            return int(candidate)
    return None


def error_reason(error: Exception, api_key: str | None = None) -> str:
    reason = getattr(error, "reason", None) or getattr(error, "message", None)
    if reason is None:
        reason = str(error)
    reason = str(reason)
    if api_key:
        reason = reason.replace(api_key, "[REDACTED]")
    return re.sub(
        r"https?://[^\s\"'<>]+",
        "[URL REDACTED]",
        reason,
        flags=re.IGNORECASE,
    )


def is_network_timeout(error: Exception) -> bool:
    return any(
        "timeout" in error_type.__name__.lower()
        for error_type in type(error).__mro__
    )


def append_failed(
    article: pd.Series,
    error: Exception,
    api_key: str | None = None,
) -> None:
    FAILED_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "url": str(article["url"]),
        "exception": type(error).__name__,
        "status": status_code(error),
        "reason": error_reason(error, api_key),
    }
    with FAILED_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_usage(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return 0, 0
    input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    return input_tokens, output_tokens


def call_gemini(
    client: Any,
    model: str,
    prompt: str,
    sleep_fn=time.sleep,
) -> tuple[str, int, int]:
    cached = read_cache(prompt, model)
    if cached is not None:
        return (
            str(cached["text"]),
            int(cached.get("input_tokens", 0)),
            int(cached.get("output_tokens", 0)),
        )

    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": ArticleLabel.model_json_schema(),
                },
            )
            text = getattr(response, "text", None)
            if text is None:
                text = str(response)
            input_tokens, output_tokens = extract_usage(response)
            write_cache(
                prompt,
                model,
                {
                    "text": text,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
            )
            return text, input_tokens, output_tokens
        except Exception as error:
            code = status_code(error)
            if code in (429, 500, 503) or is_network_timeout(error):
                if attempt == len(RETRY_DELAYS):
                    raise
                sleep_fn(RETRY_DELAYS[attempt])
                continue
            raise

    raise RuntimeError("Gemini request failed")


def parse_answer(answer: str) -> ArticleLabel:
    try:
        payload = json.loads(answer)
    except json.JSONDecodeError as error:
        raise ValidationError.from_exception_data(
            "ArticleLabel",
            [{"type": "invalid_json", "loc": (), "input": answer}],
        ) from error
    return ArticleLabel.model_validate(payload)


def label_articles(
    model: str,
    daily_limit: int = 450,
    limit: int | None = None,
    per_minute: int = 12,
    seed: int = SEED,
    client: Any | None = None,
    articles_path: Path = ARTICLES_PATH,
    labels_path: Path = LABELS_PATH,
    sleep_fn=time.sleep,
) -> pd.DataFrame:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if client is None:
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        try:
            from google import genai
        except ImportError as error:
            raise RuntimeError("google-genai is not installed") from error
        client = genai.Client(api_key=api_key)

    articles = pd.read_parquet(articles_path)
    existing = load_labels(labels_path)
    if "url" in existing.columns:
        articles = articles[~articles["url"].isin(set(existing["url"].dropna()))]

    if limit is not None:
        articles = sample_across_months(articles, limit, seed)
    else:
        articles = sample_across_months(articles, len(articles), seed)

    call_count = 0
    request_times: list[float] = []
    new_labels: list[dict[str, Any]] = []

    for _, article in articles.iterrows():
        if call_count >= daily_limit:
            break

        prompt = build_prompt(article)
        cached = read_cache(prompt, model)
        if cached is None:
            now = time.monotonic()
            request_times = [timestamp for timestamp in request_times if now - timestamp < 60]
            if len(request_times) >= per_minute:
                wait = 60 - (now - request_times[0])
                if wait > 0:
                    sleep_fn(wait)
                request_times = []

        try:
            answer, input_tokens, output_tokens = call_gemini(
                client,
                model,
                prompt,
                sleep_fn,
            )
        except Exception as error:
            code = status_code(error)
            api_key = os.getenv("GEMINI_API_KEY")
            reason = error_reason(error, api_key)
            print(
                f"Gemini API error: exception={type(error).__name__} "
                f"status={code if code is not None else 'unknown'} "
                f"reason={reason}"
            )
            append_failed(article, error, api_key)
            if cached is None:
                call_count += 1
                request_times.append(time.monotonic())
            continue

        if cached is None:
            call_count += 1
            request_times.append(time.monotonic())
            append_cost(model, input_tokens, output_tokens)

        try:
            label = parse_answer(answer)
        except (ValidationError, json.JSONDecodeError) as error:
            append_invalid(article, answer, str(error))
            continue

        row = {"url": str(article["url"]), **label.model_dump()}
        new_labels.append(row)
        combined = append_labels(new_labels, labels_path)
        print(f"Articles labeled so far: {len(combined)}")

    return append_labels([], labels_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label Guardian articles with Gemini")
    parser.add_argument("--model", required=True)
    parser.add_argument("--daily-limit", type=int, default=450)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.daily_limit < 1:
        parser.error("--daily-limit must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    label_articles(
        model=args.model,
        daily_limit=args.daily_limit,
        limit=args.limit,
    )
