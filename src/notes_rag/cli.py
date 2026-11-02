import argparse
import asyncio
import logging
import os
import signal
from datetime import datetime

from cron_converter import Cron

from notes_rag.config import Config
from notes_rag.extractor.markdown_extractor import MarkdownExtractor
from notes_rag.logging_config import setup_logging
from notes_rag.mcp_server import MCPServer
from notes_rag.storage import Storage

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Notes RAG - local retrieval augmented generation for markdown notes")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable DEBUG logging level",
    )
    parser.add_argument(
        "--rebuild-on-start",
        action="store_true",
        default=False,
        help="Trigger a VectorDB rebuild immediately on startup (default: wait for first cron slot)",
    )
    return parser.parse_args(argv)


async def run_storage_loop(
    storage: Storage, cron: str, timezone: str, rebuild_on_start: bool = False,
    extractors: list[MarkdownExtractor] | None = None, poll_interval: int = 60,
):
    if poll_interval <= 0:
        raise ValueError("Git poll interval must be positive")
    schedule = Cron(cron).schedule(timezone_str=timezone)
    next_run = datetime.now().timestamp() if rebuild_on_start else schedule.next().timestamp()
    indexed_revisions: list[str | None] | None = None
    while True:
        revisions = await asyncio.gather(*(asyncio.to_thread(e.remote_revision) for e in (extractors or [])))
        # A failed remote check must never be treated as a new commit.
        changed = bool(revisions) and all(revisions) and revisions != indexed_revisions
        if changed or datetime.now().timestamp() >= next_run:
            try:
                await storage.rebuild()
            except Exception:
                logger.exception("Index rebuild failed; will retry")
            else:
                if revisions and all(revisions):
                    local = [extractor.local_revision() for extractor in (extractors or [])]
                    if local == revisions:
                        indexed_revisions = list(revisions)
            next_run = schedule.next().timestamp()
        await asyncio.sleep(min(poll_interval, max(0, next_run - datetime.now().timestamp())))


async def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Resolve logging level: --verbose > LOG_LEVEL env > INFO default
    if args.verbose:
        log_level = "DEBUG"
    else:
        log_level = os.getenv("LOG_LEVEL", "INFO")

    logger = setup_logging(log_level)
    logger.debug("Logging configured at level: %s", log_level)

    config = Config.from_env()

    md_extractor = MarkdownExtractor(
        notes_directory=config.notes_directory,
        notes_repo_url=config.notes_repo_url,
        notes_repo_branch=config.notes_branch,
    )

    storage = Storage(config, [md_extractor])
    webserver = MCPServer(storage, host=config.server_host, port=config.server_port)

    loop = asyncio.get_running_loop()
    main_task = asyncio.current_task()

    if not main_task:
        raise RuntimeError("Something went wrong!")

    def handle_signal():
        main_task.cancel()

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGABRT):
        loop.add_signal_handler(sig, handle_signal)

    try:
        await asyncio.gather(
            run_storage_loop(
                storage, config.rebuild_cron, config.timezone, rebuild_on_start=args.rebuild_on_start,
                extractors=[md_extractor], poll_interval=int(os.getenv("GIT_POLL_INTERVAL", "60")),
            ),
            webserver.run(),
        )
    except asyncio.CancelledError:
        logger.info("Stopping application")
    finally:
        logger.info("Stopped application!")

# Used for script
def main_sync():
    asyncio.run(main())
