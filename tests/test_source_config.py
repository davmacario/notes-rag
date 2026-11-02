import json

import pytest

from notes_rag.config import Config, load_sources


def test_load_multiple_sources_and_expand_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKEN", "secret")
    config = tmp_path / "sources.json"
    config.write_text(json.dumps({"sources": [
        {"type": "github", "url": "https://{{ TOKEN }}@github.com/example/one.git", "directory": str(tmp_path / "one")},
        {"type": "github", "url": "https://github.com/example/two.git", "directory": str(tmp_path / "two"), "branch": "docs"},
    ]}))
    sources = load_sources(config)
    assert len(sources) == 2
    assert sources[0].url == "https://secret@github.com/example/one.git"
    assert sources[1].branch == "docs"
    monkeypatch.setenv("SOURCES_CONFIG", str(config))
    monkeypatch.delenv("NOTES_REPO_URL", raising=False)
    assert len(Config.from_env().sources) == 2


@pytest.mark.parametrize("source", [
    {"type": "other", "url": "https://github.com/a/b", "directory": "a"},
    {"type": "github", "url": "https://example.com/a/b", "directory": "a"},
    {"type": "github", "url": "https://{{ MISSING_SECRET }}@github.com/a/b", "directory": "a"},
])
def test_reject_invalid_source(tmp_path, monkeypatch, source):
    monkeypatch.delenv("MISSING_SECRET", raising=False)
    path = tmp_path / "sources.json"
    path.write_text(json.dumps({"sources": [source]}))
    with pytest.raises(ValueError):
        load_sources(path)
