import logging
import argparse
import os
import sys

from notes_rag.config import Config
from notes_rag.extractor.markdown_extractor import MarkdownExtractor
from notes_rag.logging_config import setup_logging
from notes_rag.storage import Storage

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Notes RAG - local retrieval augmented generation for markdown notes")
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable DEBUG logging level",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main entry point."""
    args = parse_args(argv)

    # Resolve logging level: --verbose > LOG_LEVEL env > INFO default
    if args.verbose:
        log_level = "DEBUG"
    else:
        log_level = os.getenv("LOG_LEVEL", "INFO")

    logger = setup_logging(log_level)
    logger.debug("Logging configured at level: %s", log_level)

    config = Config.from_env()

    md_extractor = MarkdownExtractor(
        notes_directory="/Users/dmacario/notes",
        notes_repo_url="https://github.com/davmacario/notes.git",
        notes_repo_branch="personal",
    )

    storage = Storage(config, [md_extractor], "./.chroma")

    try:
        while True:
            storage.rebuild()
            import time
            time.sleep(3600)
    except KeyboardInterrupt:
        logger.info("Shutting down")
        sys.exit(0)


if __name__ == "__main__":
    main()
