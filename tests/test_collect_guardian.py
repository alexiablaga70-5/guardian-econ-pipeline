from pathlib import Path

import pandas as pd

from src.collect_guardian import collect_articles


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
        assert params["show-fields"] == "bodyText"
        assert params["from-date"] == "2020-01-01"
        return MockResponse()


def test_collect_articles_mocked(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GUARDIAN_API_KEY", "test-key")
    output_path = tmp_path / "articles.parquet"

    result = collect_articles(output_path=output_path, session=MockSession())

    assert len(result) == 1
    assert set(result["url"]) == {
        "https://www.theguardian.com/business/test-article"
    }
    assert result.iloc[0]["title"] == "Test article"
    assert result.iloc[0]["body"] == "Article body text"
    saved = pd.read_parquet(output_path)
    assert len(saved) == 1
    assert saved.iloc[0]["url"] == result.iloc[0]["url"]
