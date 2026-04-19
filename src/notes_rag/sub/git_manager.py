from pathlib import Path
from typing import Optional

from git import GitCommandError, InvalidGitRepositoryError, Repo


class GitManager:
    """Manages Git repository operations for the RAG notes system."""

    def __init__(self, repo_path: Path):
        """Initialize GitManager with a repository path.

        Args:
            repo_path: Path to the local Git repository.

        Raises:
            ValueError: If the path exists but is not a valid Git repository.
        """
        self.repo_path = Path(repo_path)
        if self.repo_path.exists():
            try:
                self.repo: Optional[Repo] = Repo(self.repo_path)
            except InvalidGitRepositoryError:
                raise ValueError(f"Path {repo_path} is not a valid Git repository")
        else:
            self.repo = None

    def clone(self, url: str, branch: str = "main") -> Repo:
        """Clone a repository from a remote URL.

        Args:
            url: Remote repository URL.
            branch: Branch to checkout after cloning. Defaults to "main".

        Returns:
            The cloned Repo object.
        """
        self.repo_path.mkdir(parents=True, exist_ok=True)
        repo = Repo.clone_from(url, self.repo_path)
        repo.git.checkout(branch)
        self.repo = repo
        return repo

    def checkout(self, branch: str) -> bool:
        """Checkout a branch in the repository.

        Args:
            branch: Name of the branch to checkout.

        Returns:
            True if successful, False otherwise.
        """
        if self.repo is None:
            return False
        try:
            self.repo.git.checkout(branch)
            return True
        except GitCommandError as e:
            print(f"Failed to checkout branch: {e}")
            return False

    def fetch(self) -> bool:
        """Fetch updates from the remote repository.

        Returns:
            True if successful.

        Raises:
            AttributeError: If repository is not initialized.
        """
        if self.repo is None:
            raise AttributeError("Repository not initialized. Call clone() first.")
        self.repo.remotes.origin.fetch()
        return True

    def pull(self, rebase: bool = False) -> bool:
        """Pull latest changes from the remote repository.

        Args:
            rebase: If True, use rebase instead of merge. Defaults to False.

        Returns:
            True if successful, False otherwise.

        Raises:
            AttributeError: If repository is not initialized.
        """
        if self.repo is None:
            raise AttributeError("Repository not initialized. Call clone() first.")
        try:
            if rebase:
                self.repo.git.pull("origin", "--rebase")
            else:
                self.repo.git.pull("origin")
            return True
        except GitCommandError as e:
            print(f"Pull failed: {e}")
            return False

    def status(self) -> str:
        """Get the repository status.

        Returns:
            Git status string.

        Raises:
            AttributeError: If repository is not initialized.
        """
        if self.repo is None:
            raise AttributeError("Repository not initialized. Call clone() first.")
        return self.repo.git.status()
