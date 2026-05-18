import argparse
import signal
import asyncio
import os
from pathlib import Path

from notes_rag.config import Config
from notes_rag.extractor.markdown_extractor import MarkdownExtractor
from notes_rag.logging_config import setup_logging
from notes_rag.storage import Storage
from notes_rag.webserver import MCPServer


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
    return parser.parse_args(argv)


async def run_storage_loop(storage: Storage, cron: str):
    while True:
        await asyncio.to_thread(storage.rebuild)
        # TODO: evaluate cron string

        await asyncio.sleep(3600)


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

    # FIXME: avoid hardcoding
    md_extractor = MarkdownExtractor(
        notes_directory=Path("/Users/dmacario/notes"),
        notes_repo_url="https://github.com/davmacario/notes.git",
        notes_repo_branch="personal",
    )

    storage = Storage(config, [md_extractor])
    webserver = MCPServer(storage)

    loop = asyncio.get_running_loop()
    main_task = asyncio.current_task()

    if not main_task:
        raise RuntimeError("Something went wrong!")

    def handle_signal():
        main_task.cancel()

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGABRT):
        loop.add_signal_handler(sig, handle_signal)

    try:
        await asyncio.gather(run_storage_loop(storage, config.rebuild_cron), webserver.run())
    except asyncio.CancelledError:
        logger.info("Stopping application")
    finally:
        # TODO: cleanup resources
        logger.info("Stopped application!")
