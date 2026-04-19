from unittest.mock import MagicMock

import pytest
from git import GitCommandError, InvalidGitRepositoryError

from notes_rag.sub.git_manager import GitManager


class TestGitManager:
    def test_init_valid_repository(self, tmp_path, monkeypatch):
        repo_path = tmp_path / "test_repo"
        repo_path.mkdir()
        mock_repo = MagicMock()
        monkeypatch.setattr("notes_rag.sub.git_manager.Repo", mock_repo)
        manager = GitManager(repo_path)
        assert manager.repo_path == repo_path
        mock_repo.assert_called_once_with(repo_path)

    def test_init_invalid_repository(self, tmp_path, monkeypatch):
        repo_path = tmp_path / "invalid_repo"
        repo_path.mkdir()
        original_repo = MagicMock(side_effect=InvalidGitRepositoryError())
        monkeypatch.setattr("notes_rag.sub.git_manager.Repo", original_repo)
        with pytest.raises(ValueError, match="is not a valid Git repository"):
            GitManager(repo_path)

    def test_init_new_path_no_repo(self, tmp_path):
        repo_path = tmp_path / "new_repo"
        manager = GitManager(repo_path)
        assert manager.repo_path == repo_path
        assert manager.repo is None

    def test_clone_success(self, tmp_path, monkeypatch):
        mock_clone = MagicMock()
        monkeypatch.setattr("notes_rag.sub.git_manager.Repo.clone_from", mock_clone)
        manager = GitManager(tmp_path / "test")
        mock_repo = MagicMock()
        mock_clone.return_value = mock_repo
        repo = manager.clone("https://github.com/user/repo.git", "main")
        assert repo == mock_repo
        mock_repo.git.checkout.assert_called_once_with("main")

    def test_clone_creates_parent_dirs(self, tmp_path, monkeypatch):
        mock_clone = MagicMock()
        monkeypatch.setattr("notes_rag.sub.git_manager.Repo.clone_from", mock_clone)
        manager = GitManager(tmp_path / "nested" / "deep" / "repo")
        mock_repo = MagicMock()
        mock_clone.return_value = mock_repo
        manager.clone("https://github.com/user/repo.git")
        assert (tmp_path / "nested" / "deep" / "repo").exists()

    def test_clone_with_custom_branch(self, tmp_path, monkeypatch):
        mock_clone = MagicMock()
        monkeypatch.setattr("notes_rag.sub.git_manager.Repo.clone_from", mock_clone)
        manager = GitManager(tmp_path / "test")
        mock_repo = MagicMock()
        mock_clone.return_value = mock_repo
        manager.clone("https://github.com/user/repo.git", "develop")
        mock_repo.git.checkout.assert_called_once_with("develop")

    def test_checkout_success(self, tmp_path, monkeypatch):
        mock_repo = MagicMock()
        manager = GitManager(tmp_path / "test")
        manager.repo = mock_repo
        monkeypatch.setattr(manager.repo.git, "checkout", MagicMock())
        result = manager.checkout("feature-branch")
        assert result is True
        manager.repo.git.checkout.assert_called_once_with("feature-branch")

    def test_checkout_failure(self, tmp_path, capsys):
        mock_repo = MagicMock()
        manager = GitManager(tmp_path / "test")
        manager.repo = mock_repo
        mock_repo.git.checkout.side_effect = GitCommandError("checkout", "fatal: invalid branch")
        result = manager.checkout("invalid-branch")
        assert result is False
        captured = capsys.readouterr()
        assert "Failed to checkout branch" in captured.out

    def test_checkout_no_repo(self, tmp_path):
        manager = GitManager(tmp_path / "test")
        result = manager.checkout("feature-branch")
        assert result is False

    def test_fetch_success(self, tmp_path):
        mock_repo = MagicMock()
        manager = GitManager(tmp_path / "test")
        manager.repo = mock_repo
        mock_repo.remotes.origin.fetch = MagicMock()
        result = manager.fetch()
        assert result is True
        mock_repo.remotes.origin.fetch.assert_called_once()

    def test_fetch_no_repo(self, tmp_path):
        manager = GitManager(tmp_path / "test")
        with pytest.raises(AttributeError):
            manager.fetch()

    def test_pull_success(self, tmp_path):
        mock_repo = MagicMock()
        manager = GitManager(tmp_path / "test")
        manager.repo = mock_repo
        result = manager.pull()
        assert result is True
        mock_repo.git.pull.assert_called_once_with("origin")

    def test_pull_with_rebase(self, tmp_path):
        mock_repo = MagicMock()
        manager = GitManager(tmp_path / "test")
        manager.repo = mock_repo
        result = manager.pull(rebase=True)
        assert result is True
        mock_repo.git.pull.assert_called_once_with("origin", "--rebase")

    def test_pull_failure(self, tmp_path, capsys):
        mock_repo = MagicMock()
        manager = GitManager(tmp_path / "test")
        manager.repo = mock_repo
        mock_repo.git.pull.side_effect = GitCommandError("pull", "fatal: failed")
        result = manager.pull()
        assert result is False
        captured = capsys.readouterr()
        assert "Pull failed" in captured.out

    def test_status(self, tmp_path):
        mock_repo = MagicMock()
        mock_repo.git.status.return_value = "On branch main"
        manager = GitManager(tmp_path / "test")
        manager.repo = mock_repo
        result = manager.status()
        assert result == "On branch main"
        mock_repo.git.status.assert_called_once()
