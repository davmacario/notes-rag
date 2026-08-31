import logging
from pathlib import Path
from typing import Generator, List

import git
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import TextNode
from llama_index.readers.file import MarkdownReader

from notes_rag.extractor.abstract import BaseExtractor
from notes_rag.sub.git_manager import GitManager

logger = logging.getLogger(__name__)


# TODO: figure out a way to remove YAML header from markdown files that have them


class MarkdownExtractor(BaseExtractor):
    def __init__(
        self,
        notes_directory: Path,
        notes_repo_url: str,
        notes_repo_branch: str | None = None,
    ):
        """
        Args:
            notes_directory: path to the local copy of the notes
            notes_repo_url: Git URL of the repo. If private, it should include an access token
            notes_repo_branch: branch of the Git repo
        """
        self.notes_directory = notes_directory.resolve()
        self._git = GitManager(self.notes_directory)
        self._notes_repo_url = notes_repo_url
        self._notes_repo_branch = notes_repo_branch

    def _init_or_restore_notes_repo(self):
        """Initialize or restore the repository containing the notes.

        Clones the repository if the notes directory doesn't exist yet.
        If the directory already exists, assumes it's a valid Git repo and skips cloning.
        """
        if not self._notes_repo_url:
            raise ValueError("notes_repo_url is required for cloning")

        if not self._git.repo_path.exists():
            if self._notes_repo_branch:
                self._git.clone(self._notes_repo_url, self._notes_repo_branch)
            else:
                self._git.clone(self._notes_repo_url)
        elif self._notes_repo_branch:
            self._git.checkout(self._notes_repo_branch)

    def _fetch_latest_notes(self):
        """Fetch the latest version of the notes (Git).

        Returns:
            True if fetch and pull succeed, False otherwise.
        """
        if not self._git.repo_path.exists():
            return False

        if not self._git.fetch():
            return False

        return self._git.pull()

    def get_nodes(self, batch_size: int | None = None) -> Generator[List[TextNode], None, None]:
        self._init_or_restore_notes_repo()
        try:
            self._fetch_latest_notes()
        except git.GitCommandError:
            # This will happen if mounting a local clone of a private git repo on the container (unless valid
            # credentials are provided)
            # If here, it means that the directory is available locally.
            logger.error("Unable to fetch latest changes! Will use existing copy of the repository.")

        if not self.notes_directory.exists():
            raise FileNotFoundError("Local copy of repository was not found!")

        md_files: List[Path] = []
        for file_path in self.notes_directory.rglob("*.md"):
            if file_path.is_file():
                md_files.append(file_path)

        if not md_files:
            logger.warning("No Markdown documents were found!")
            return

        md_reader = MarkdownReader(remove_hyperlinks=True, remove_images=True)
        node_parser = MarkdownNodeParser()

        count_nodes = 0
        return_chunk = []
        for file_path in md_files:
            file_docs = md_reader.load_data(str(file_path))
            nodes = node_parser.get_nodes_from_documents(file_docs)

            # Add enough nodes to
            for node in nodes:
                count_nodes += 1
                node.metadata["source_file"] = str(file_path.relative_to(self.notes_directory))
                return_chunk.append(node)

                if batch_size and len(return_chunk) >= batch_size:
                    yield return_chunk
                    return_chunk = []
        yield return_chunk
        logger.debug(f"MarkdownExtractor extracted {count_nodes} nodes from {len(md_files)} files")
