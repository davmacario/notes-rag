import logging
from pathlib import Path
from typing import Optional

from git import GitCommandError, InvalidGitRepositoryError, Repo


class GitManager:
    """Manages Git repository operations for the RAG notes system."""

    def __init__(self, repo_path: Path):
        self.repo_path = Path(repo_path)
        if self.repo_path.exists():
            try:
                self.repo: Optional[Repo] = Repo(self.repo_path)
            except InvalidGitRepositoryError:
                raise ValueError(f"Path {repo_path} is not a valid Git repository")
        else:
            self.repo = None

    def clone(self, url: str, branch: str = "main") -> Repo:
        self.repo_path.mkdir(parents=True, exist_ok=True)
        repo = Repo.clone_from(url, self.repo_path)
        repo.git.checkout(branch)
        self.repo = repo
        return repo

    def checkout(self, branch: str) -> bool:
        if self.repo is None:
            return False
        try:
            self.repo.git.checkout(branch)
            return True
        except GitCommandError as e:
            logger = logging.getLogger(__name__)
            logger.error("Failed to checkout branch: %s", e)
            return False

    def fetch(self) -> bool:
        if self.repo is None:
            raise AttributeError("Repository not initialized. Call clone() first.")
        self.repo.remotes.origin.fetch()
        return True

    def pull(self, rebase: bool = False) -> bool:
        if self.repo is None:
            raise AttributeError("Repository not initialized. Call clone() first.")
        try:
            if rebase:
                self.repo.git.pull("origin", "--rebase")
            else:
                self.repo.git.pull("origin")
            return True
        except GitCommandError as e:
            logger = logging.getLogger(__name__)
            logger.error("Pull failed: %s", e)
            return False

    def status(self) -> str:
        if self.repo is None:
            raise AttributeError("Repository not initialized. Call clone() first.")
        return self.repo.git.status()
