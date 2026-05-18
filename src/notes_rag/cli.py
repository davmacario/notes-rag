import argparse
from datetime import datetime
import logging
import signal
import asyncio
import os
from pathlib import Path

from cron_converter import Cron
from notes_rag.config import Config
from notes_rag.extractor.markdown_extractor import MarkdownExtractor
from notes_rag.logging_config import setup_logging
from notes_rag.storage import Storage
from notes_rag.mcp_server import MCPServer

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
    return parser.parse_args(argv)


async def run_storage_loop(storage: Storage, cron: str, timezone: str):
    schedule = Cron(cron).schedule(timezone_str=timezone)
    while True:
        await storage.rebuild()
        next_run_datetime = schedule.next()
        logger.info(f"Next run scheduled for {next_run_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        sleep_duration = next_run_datetime.timestamp() - datetime.now().timestamp()
        await asyncio.sleep(sleep_duration)


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
        notes_repo_url=config.notes_repo_url,
        notes_repo_branch=config.notes_branch,
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
        await asyncio.gather(run_storage_loop(storage, config.rebuild_cron, config.timezone), webserver.run())
    except asyncio.CancelledError:
        logger.info("Stopping application")
    finally:
        # TODO: cleanup resources
        logger.info("Stopped application!")
