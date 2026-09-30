import logging

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


def create_embed_model(model_name: str) -> FastEmbedEmbedding:
    """Create a FastEmbed embedding model whose tokenizer pads each batch consistently.

    Args:
        model_name: Name of the FastEmbed-supported model.

    Returns:
        FastEmbedEmbedding instance.
    """
    embed_model = FastEmbedEmbedding(model_name=model_name)
    tokenizer: Tokenizer | None = getattr(embed_model._model.model, "tokenizer", None)  # pyright: ignore[reportPrivateUsage]
    if tokenizer is None:
        raise RuntimeError(f"Embedding model {model_name!r} has no tokenizer loaded")
    enable_batch_longest_padding(tokenizer)
    return embed_model
