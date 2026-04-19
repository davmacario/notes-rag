from pathlib import Path
from threading import Thread

from notes_rag.storage import Storage
from notes_rag.sub.git_manager import GitManager

# TODO
# @dataclass
# class IndexerConfig:


class Indexer(Thread):
    """
    The Indexer takes the raw notes (markdown), chunks them, and forwards them to the Storage so that they are tracked
    in the DB and can be queried.
    It forces rebuild of the Storage on a schedule and it works in parallel to the app.

    The notes are found in a local directory
    """

    def __init__(
        self, storage: Storage, schedule: str, notes_directory: Path, notes_repo_url: str, notes_repo_branch: str
    ):
        """
        Args:
            storage: the Storage object used
            schedule: string in Cron format indicating the schedule to rebuild the DB
            notes_directory: path to the local copy of the notes
            notes_repo_url: Git URL of the repo. If private, it should include an access token
            notes_repo_branch: branch of the Git repo
        """
        self._storage = storage
        self._schedule = schedule

        self._git = GitManager(notes_directory)
        self._notes_repo_url = notes_repo_url
        self._notes_repo_branch = notes_repo_branch

    def _init_or_restore_notes_repo(self):
        """Initialize or restore the repository containing the notes"""
        pass

    def _fetch_latest_notes(self):
        """Fetch the latest version of the notes (Git)"""
        pass

    def _rebuild_db(self):
        pass

    def run(self):
        """Main loop (schedule)"""
        self._init_or_restore_notes_repo()

        while True:  # TODO: figure out schedule
            self._fetch_latest_notes()
            self._rebuild_db()

            # time.sleep()
