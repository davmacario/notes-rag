import logging
from unittest.mock import MagicMock

import pytest
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from notes_rag.sub.embedding_utils import create_embed_model, enable_batch_longest_padding


@pytest.fixture
def fixed_padding_tokenizer() -> Tokenizer:
    """Tokenizer shipping fixed-length padding shorter than its truncation length."""
    tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "a": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()  # type: ignore[assignment]
    tokenizer.enable_truncation(max_length=8)
    tokenizer.enable_padding(length=4, pad_id=0, pad_token="[PAD]")
    return tokenizer


class TestEnableBatchLongestPadding:
    """Test enable_batch_longest_padding function."""

    def test_fixed_padding_produces_ragged_batch(self, fixed_padding_tokenizer: Tokenizer):
        """Sanity check: reproduces the bug this helper fixes."""
        encoded = fixed_padding_tokenizer.encode_batch(["a", "a a a a a a"])
        assert [len(e.ids) for e in encoded] == [4, 6]

    def test_pads_batch_to_longest(self, fixed_padding_tokenizer: Tokenizer, caplog):
        with caplog.at_level(logging.DEBUG):
            enable_batch_longest_padding(fixed_padding_tokenizer)

        encoded = fixed_padding_tokenizer.encode_batch(["a", "a a a a a a"])
        assert [len(e.ids) for e in encoded] == [6, 6]
        assert fixed_padding_tokenizer.padding is not None
        assert fixed_padding_tokenizer.padding["length"] is None
        assert fixed_padding_tokenizer.padding["pad_id"] == 0
        assert fixed_padding_tokenizer.padding["pad_token"] == "[PAD]"
        assert "fixed-length padding" in caplog.text

    def test_keeps_truncation(self, fixed_padding_tokenizer: Tokenizer):
        enable_batch_longest_padding(fixed_padding_tokenizer)

        encoded = fixed_padding_tokenizer.encode_batch(["a " * 20])
        assert len(encoded[0].ids) == 8

    def test_leaves_dynamic_padding_untouched(self, fixed_padding_tokenizer: Tokenizer):
        fixed_padding_tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        before = fixed_padding_tokenizer.padding

        enable_batch_longest_padding(fixed_padding_tokenizer)

        assert fixed_padding_tokenizer.padding == before

    def test_no_padding_is_noop(self, fixed_padding_tokenizer: Tokenizer):
        fixed_padding_tokenizer.no_padding()

        enable_batch_longest_padding(fixed_padding_tokenizer)

        assert fixed_padding_tokenizer.padding is None


class TestCreateEmbedModel:
    """Test create_embed_model function."""

    def test_creates_model_and_fixes_padding(self, monkeypatch, fixed_padding_tokenizer: Tokenizer):
        mocked_embedding = MagicMock(spec=FastEmbedEmbedding)
        mocked_embedding._model = MagicMock()
        mocked_embedding._model.model.tokenizer = fixed_padding_tokenizer
        mock_factory = MagicMock(return_value=mocked_embedding)
        monkeypatch.setattr("notes_rag.sub.embedding_utils.FastEmbedEmbedding", mock_factory)

        embed_model = create_embed_model("some/model")

        mock_factory.assert_called_once_with(model_name="some/model")
        assert embed_model == mocked_embedding
        assert fixed_padding_tokenizer.padding is not None
        assert fixed_padding_tokenizer.padding["length"] is None

    def test_missing_tokenizer_raises(self, monkeypatch):
        mocked_embedding = MagicMock(spec=FastEmbedEmbedding)
        mocked_embedding._model = MagicMock()
        mocked_embedding._model.model.tokenizer = None
        monkeypatch.setattr(
            "notes_rag.sub.embedding_utils.FastEmbedEmbedding", MagicMock(return_value=mocked_embedding)
        )

        with pytest.raises(RuntimeError, match="tokenizer"):
            create_embed_model("some/model")
