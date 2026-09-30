import asyncio
import logging
from typing import Any

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)


def enable_batch_longest_padding(tokenizer: Tokenizer) -> None:
    """Replace fixed-length padding with padding to the longest sequence in each batch.

    fastembed keeps the padding shipped in `tokenizer.json` when present. If that is a fixed length shorter than the
    truncation length, longer inputs stay unpadded and batches become ragged, making ONNX input creation fail.

    Args:
        tokenizer: Tokenizer to update in place.
    """
    padding = tokenizer.padding
    if not padding or padding["length"] is None:
        return
    logger.debug(f"Replacing tokenizer fixed-length padding ({padding['length']}) with batch-longest padding")
    tokenizer.enable_padding(
        direction=padding["direction"],
        pad_id=padding["pad_id"],
        pad_type_id=padding["pad_type_id"],
        pad_token=padding["pad_token"],
        pad_to_multiple_of=padding["pad_to_multiple_of"],
    )


class BoundedFastEmbedEmbedding(FastEmbedEmbedding):
    """FastEmbedEmbedding limiting the number of text embedding batches computed concurrently by async calls.

    LlamaIndex embeds all batches of an async call at once unless `num_workers > 1`, so the limit is enforced here.
    """

    _semaphore: asyncio.Semaphore = PrivateAttr()

    def __init__(self, max_concurrency: int, **kwargs: Any) -> None:
        if max_concurrency < 1:
            raise ValueError(f"num_workers must be at least 1, got {max_concurrency}")
        super().__init__(**kwargs)
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        async with self._semaphore:
            return await super()._aget_text_embeddings(texts)


def create_embed_model(model_name: str, num_workers: int) -> FastEmbedEmbedding:
    """Create a FastEmbed embedding model whose tokenizer pads each batch consistently.

    Args:
        model_name: Name of the FastEmbed-supported model.
        num_workers: Maximum number of embedding batches computed concurrently by async calls.

    Returns:
        FastEmbedEmbedding instance.
    """
    embed_model = BoundedFastEmbedEmbedding(max_concurrency=num_workers, model_name=model_name)
    tokenizer: Tokenizer | None = getattr(embed_model._model.model, "tokenizer", None)  # pyright: ignore[reportPrivateUsage]
    if tokenizer is None:
        raise RuntimeError(f"Embedding model {model_name!r} has no tokenizer loaded")
    enable_batch_longest_padding(tokenizer)
    return embed_model
