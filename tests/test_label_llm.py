import json
from pathlib import Path

import pandas as pd
import pytest

from src.label_llm import (
    build_prompt,
    call_gemini,
    label_articles,
    sample_across_months,
)
import src.label_llm as label_module


class MockUsage:
    prompt_token_count = 10
    candidates_token_count = 5


class MockResponse:
    text = json.dumps(
        {
            "tema": "inflatie",
            "sentiment": "negativ",
            "entitati": ["Banca Centrala"],
            "perspectiva_economica": "in_scadere",
            "incredere": 0.8,
        }
    )
    usage_metadata = MockUsage()


class MockModels:
    def generate_content(self, model, contents, config):
        assert model
        assert "response_mime_type" in config
        assert "ArticleLabel" in str(config["response_json_schema"])
        return MockResponse()


class MockClient:
    models = MockModels()


@pytest.fixture(autouse=True)
def isolate_label_outputs(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(label_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(label_module, "COSTS_PATH", tmp_path / "costs.csv")
    monkeypatch.setattr(label_module, "INVALID_PATH", tmp_path / "invalid.jsonl")


def test_label_articles_mocked(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GEMINI_API_KEY", "test-secret")
    articles_path = tmp_path / "articles.parquet"
    labels_path = tmp_path / "labels.parquet"
    pd.DataFrame(
        [
            {
                "url": "https://example.test/1",
                "title": "Inflatie",
                "section": "Business",
                "date": "2026-01-10T00:00:00Z",
                "body": "Preturi in crestere.",
            },
            {
                "url": "https://example.test/2",
                "title": "Energie",
                "section": "Business",
                "date": "2026-02-10T00:00:00Z",
                "body": "Pretul energiei.",
            },
        ]
    ).to_parquet(articles_path, index=False)

    result = label_articles(
        model="test-model",
        limit=2,
        daily_limit=450,
        client=MockClient(),
        articles_path=articles_path,
        labels_path=labels_path,
        sleep_fn=lambda _: None,
    )

    assert len(result) == 2
    assert set(result["url"]) == {
        "https://example.test/1",
        "https://example.test/2",
    }


def test_cache_avoids_second_api_call(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GEMINI_API_KEY", "test-secret")
    monkeypatch.setattr(label_module, "CACHE_DIR", tmp_path / "cache")

    class CountingModels:
        calls = 0

        def generate_content(self, model, contents, config):
            self.calls += 1
            return MockResponse()

    class CountingClient:
        models = CountingModels()

    client = CountingClient()
    first = call_gemini(client, "cache-model", "same prompt", lambda _: None)
    second = call_gemini(client, "cache-model", "same prompt", lambda _: None)

    assert first[0] == second[0]
    assert client.models.calls == 1


def test_api_key_not_in_error_output(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret")
    articles_path = tmp_path / "articles.parquet"
    labels_path = tmp_path / "labels.parquet"
    pd.DataFrame(
        [
            {
                "url": "https://example.test/error",
                "title": "Error",
                "section": "Business",
                "date": "2026-01-10T00:00:00Z",
                "body": "Text.",
            }
        ]
    ).to_parquet(articles_path, index=False)

    class ErrorClient:
        class Models:
            def generate_content(self, model, contents, config):
                raise RuntimeError("https://example.test/super-secret")

        models = Models()

    label_articles(
        model="error-model",
        limit=1,
        client=ErrorClient(),
        articles_path=articles_path,
        labels_path=labels_path,
        sleep_fn=lambda _: None,
    )

    message = capsys.readouterr().out
    assert "super-secret" not in message


def test_sample_spans_full_date_range_and_shuffles_deterministically():
    articles = pd.DataFrame(
        [
            {
                "url": f"https://example.test/{year}-{month:02d}",
                "title": f"Article {year}-{month:02d}",
                "date": f"{year}-{month:02d}-15T00:00:00Z",
            }
            for year in range(2020, 2027)
            for month in range(1, 13)
        ]
    )

    sample = sample_across_months(articles, limit=7, seed=42)
    repeat = sample_across_months(articles, limit=7, seed=42)
    selected_dates = pd.to_datetime(sample["date"], utc=True)

    assert sample["url"].tolist() == repeat["url"].tolist()
    assert selected_dates.min().year == 2020
    assert selected_dates.max().year == 2026
    assert not selected_dates.is_monotonic_increasing


def test_daily_limit_processes_shuffled_sample_across_years(monkeypatch, tmp_path: Path):
    articles_path = tmp_path / "articles.parquet"
    labels_path = tmp_path / "labels.parquet"
    source = pd.DataFrame(
        [
            {
                "url": f"https://example.test/{year}-{month:02d}",
                "title": f"Article {year}-{month:02d}",
                "section": "Business",
                "date": f"{year}-{month:02d}-15T00:00:00Z",
                "body": "Economic report.",
            }
            for year in range(2020, 2027)
            for month in range(1, 13)
        ]
    )
    source.to_parquet(articles_path, index=False)

    class RecordingModels:
        def __init__(self):
            self.titles = []

        def generate_content(self, model, contents, config):
            self.titles.append(contents.splitlines()[1])
            return MockResponse()

    class RecordingClient:
        def __init__(self):
            self.models = RecordingModels()

    client = RecordingClient()
    label_articles(
        model="partial-run-model",
        daily_limit=12,
        client=client,
        articles_path=articles_path,
        labels_path=labels_path,
        seed=42,
        sleep_fn=lambda _: None,
    )

    years = {
        int(title.rsplit(" ", 1)[1][:4])
        for title in client.models.titles
    }
    assert len(client.models.titles) == 12
    assert len(years) >= 4
    assert client.models.titles != sorted(client.models.titles)


def test_prompt_marks_non_economic_articles_as_other():
    prompt = build_prompt(
        pd.Series(
            {
                "title": "Music festival",
                "body": "A report about a music festival.",
            }
        )
    )

    assert '"altele"' in prompt
    assert '"neclar"' in prompt
    assert "muzica" in prompt
