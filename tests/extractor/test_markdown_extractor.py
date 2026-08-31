import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.readers.file import MarkdownReader

from notes_rag.extractor.markdown_extractor import MarkdownExtractor
from notes_rag.sub.git_manager import GitManager


class TestMarkdownExtractor:
    @pytest.fixture
    def mocked_git_manager(self):
        mock_gm = MagicMock(spec=GitManager)
        return mock_gm

    @pytest.fixture(autouse=True)
    def mock_git_manager(self, monkeypatch, mocked_git_manager):
        monkeypatch.setattr(
            "notes_rag.extractor.markdown_extractor.GitManager", MagicMock(return_value=mocked_git_manager)
        )

    @pytest.fixture
    def mocked_md_reader(self):
        return MagicMock(spec=MarkdownReader)

    @pytest.fixture
    def mocked_md_node_parser(self):
        return MagicMock(spec=MarkdownNodeParser)

    @pytest.fixture
    def mocked_md_files(self, tmp_path: Path):
        notes_dir = tmp_path / "notes_cache"
        notes_dir.mkdir(exist_ok=True)
        for i in range(5):
            fname = f"file_{i}.md"
            fpath = notes_dir / fname
            with fpath.open("w") as fp:
                fp.write(f"# FILE {i}\n\nHello from {fname}!")

        return notes_dir

    @pytest.fixture
    def md_extractor(self, mocked_md_files):
        md_ext = MarkdownExtractor(mocked_md_files, "https://github.com/test/notes", "main")
        md_ext._init_or_restore_notes_repo = MagicMock()
        md_ext._fetch_latest_notes = MagicMock()
        return md_ext

    # ---

    def test_init(self, tmp_path, mocked_git_manager):
        notes_dir = tmp_path / "notes_cache"
        notes_repo_url = "https://github.com/test/notes"
        notes_branch = "my-feat-branch"

        md_ext = MarkdownExtractor(notes_dir, notes_repo_url, notes_branch)

        assert md_ext.notes_directory == notes_dir.resolve()
        assert md_ext._git == mocked_git_manager
        assert md_ext._notes_repo_url == notes_repo_url
        assert md_ext._notes_repo_branch == notes_branch

    def test_get_nodes(self, caplog, monkeypatch, md_extractor):

        with caplog.at_level(logging.DEBUG):
            out = list(md_extractor.get_nodes(batch_size=2))

        assert len(out) == 3
        assert len(out[-1]) == 1
