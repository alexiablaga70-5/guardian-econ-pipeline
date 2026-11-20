from pathlib import Path

import pandas as pd

import requests

from src.collect_guardian import collect_articles, fetch_articles


class MockResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {
            "response": {
                "pages": 1,
                "results": [
                    {
                        "webUrl": "https://www.theguardian.com/business/test-article",
                        "webTitle": "Test article",
                        "sectionName": "Business",
                        "webPublicationDate": "2026-01-10T12:00:00Z",
                        "fields": {"bodyText": "Article body text"},
                    }
                ],
            }
        }


class MockSession:
    def get(self, url, params, timeout):
        assert url.endswith("/search")
        assert params["show-fields"] == "bodyText"
        assert params["section"] == "business"
        assert params["tag"] == "business/economics"
        assert params["from-date"] == "2020-01-01"
        assert params["to-date"] == "2020-01-31"
        return MockResponse()


def test_collect_articles_mocked(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GUARDIAN_API_KEY", "test-key")
    output_path = tmp_path / "articles.parquet"

    result = collect_articles(
        output_path=output_path,
        session=MockSession(),
        max_articles=1,
        per_month=1,
        tag="business/economics",
    )

    assert len(result) == 1
    assert set(result["url"]) == {
        "https://www.theguardian.com/business/test-article"
    }
    assert result.iloc[0]["title"] == "Test article"
    assert result.iloc[0]["body"] == "Article body text"
    saved = pd.read_parquet(output_path)
    assert len(saved) == 1
    assert saved.iloc[0]["url"] == result.iloc[0]["url"]


def test_http_error_does_not_include_api_key(capsys):
    class ErrorResponse:
        status_code = 403
        url = "https://content.guardianapis.com/search?api-key=super-secret"

    class ErrorSession:
        def get(self, url, params, timeout):
            error = requests.HTTPError(
                "403 Client Error: secret=super-secret",
                response=ErrorResponse(),
            )
            raise error

    articles = fetch_articles(
        "super-secret",
        "business",
        pd.Timestamp("2020-01-01").date(),
        session=ErrorSession(),
        max_articles=1,
        tag="business/economics",
    )

    assert articles == []
    message = capsys.readouterr().out
    assert "403" in message
    assert "/search" in message
    assert "super-secret" not in message
