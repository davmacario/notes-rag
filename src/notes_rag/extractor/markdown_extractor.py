import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.readers.file import MarkdownReader

from notes_rag.extractor.abstract import BaseExtractor, ExtractorResult
from notes_rag.sub.git_manager import GitManager

logger = logging.getLogger(__name__)


# TODO: fix configurability - later
@dataclass
class MarkdownExtractorConfig:
    """Configuration for the MarkdownExtractor."""

    chunk_size: int = 1000
    chunk_overlap: int = 200


class MarkdownExtractor(BaseExtractor):
    """
    The MarkdownExtractor takes the raw notes (markdown), chunks them, and forwards them to the Storage so that they are tracked
    in the DB and can be queried.
    It forces rebuild of the Storage on a schedule and it works in parallel to the app.

    The notes are found in a local directory
    """

    def __init__(
        self,
        notes_directory: Path,
        notes_repo_url: str,
        notes_repo_branch: str | None = None,
        config: MarkdownExtractorConfig | None = None,
    ):
        """
        Args:
            notes_directory: path to the local copy of the notes
            notes_repo_url: Git URL of the repo. If private, it should include an access token
            notes_repo_branch: branch of the Git repo
            notes_cache_dir: path to the cache directory where notes are stored
            config: optional MarkdownExtractorConfig for chunking parameters
        """
        self._config = config or MarkdownExtractorConfig()

        self.notes_directory = notes_directory.resolve()
        self._git = GitManager(self.notes_directory)
        self._notes_repo_url = notes_repo_url
        self._notes_repo_branch = notes_repo_branch

    # TODO: figure out better approach - git operations are blocking...
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

    def get_nodes(self) -> ExtractorResult:
        """Process markdown files and return nodes ready to be submitted for embedding by Storage.

        Walks through the notes directory recursively, parses .md files, chunks them,
        and returns the TextNodes. Storage will handle embedding and storing.

        Args:
            storage: The Storage instance to use for getting the chroma_path.

        Returns:
            ExtractorResult object
        """
        self._init_or_restore_notes_repo()
        self._fetch_latest_notes()

        if not self.notes_directory.exists():
            return ExtractorResult(nodes=[], files_processed=0)

        md_files: List[Path] = []
        for file_path in self.notes_directory.rglob("*.md"):
            if file_path.is_file():
                md_files.append(file_path)

        if not md_files:
            return ExtractorResult(nodes=[], files_processed=0)

        md_reader = MarkdownReader(remove_hyperlinks=True, remove_images=True)
        node_parser = MarkdownNodeParser()

        all_nodes = []
        for file_path in md_files:
            file_docs = md_reader.load_data(str(file_path))
            nodes = node_parser.get_nodes_from_documents(file_docs)
            for node in nodes:
                node.metadata["source_file"] = str(file_path.relative_to(self.notes_directory))
            all_nodes.extend(nodes)

        return ExtractorResult(nodes=all_nodes, files_processed=len(md_files))
