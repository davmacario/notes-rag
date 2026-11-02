import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from notes_rag.cli import run_storage_loop


async def test_rebuilds_on_changed_remote_revision(monkeypatch):
    storage = MagicMock()
    storage.rebuild = AsyncMock()
    extractor = MagicMock()
    extractor.remote_revision.side_effect = ["first", "first", "second", "second"]
    extractor.local_revision.side_effect = ["first", "second"]
    calls = 0

    async def sleep(_):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise asyncio.CancelledError

    monkeypatch.setattr("notes_rag.cli.asyncio.sleep", sleep)
    with pytest.raises(asyncio.CancelledError):
        await run_storage_loop(storage, "0 0 * * *", "UTC", extractors=[extractor])
    assert storage.rebuild.await_count == 2


async def test_failed_check_does_not_trigger_rebuild(monkeypatch):
    storage = MagicMock()
    storage.rebuild = AsyncMock()
    extractor = MagicMock()
    extractor.remote_revision.return_value = None
    monkeypatch.setattr("notes_rag.cli.asyncio.sleep", AsyncMock(side_effect=asyncio.CancelledError))
    with pytest.raises(asyncio.CancelledError):
        await run_storage_loop(storage, "0 0 * * *", "UTC", extractors=[extractor])
    storage.rebuild.assert_not_awaited()
