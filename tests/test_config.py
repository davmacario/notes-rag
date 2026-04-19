"""Tests for config module."""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from notes_rag.config import Config, DEFAULT_CONFIG


class TestConfig:
    """Test Config class initialization and defaults."""

    def test_defaults_are_set(self):
        """Test that default values match specification."""
        import os

        # Set required env vars for this test
        os.environ["OLLAMA_HOST"] = "http://localhost:11434"
        os.environ["NOTES_REPO_URL"] = "https://test@github.com/test/repo.git"

        try:
            config = Config()
            assert config.ollama_host == "http://localhost:11434"
            assert config.ollama_model == "mistral"
            assert config.notes_repo_url == "https://test@github.com/test/repo.git"
            assert config.notes_cache_dir == "./notes-cache"
            assert config.chunk_size == 1000
            assert config.chunk_overlap == 200
            assert config.top_k == 5
        finally:
            # Clean up
            os.environ.pop("OLLAMA_HOST", None)
            os.environ.pop("NOTES_REPO_URL", None)

    def test_requires_required_env_vars(self):
        """Test that Config requires OLLAMA_HOST and NOTES_REPO_URL."""
        import os

        # Save original values
        original_host = os.environ.get("OLLAMA_HOST")
        original_repo = os.environ.get("NOTES_REPO_URL")

        try:
            # Remove required env vars
            os.environ.pop("OLLAMA_HOST", None)
            os.environ.pop("NOTES_REPO_URL", None)

            # Should raise error when required vars missing
            try:
                Config()
                assert False, "Expected ValueError to be raised"
            except ValueError as e:
                assert "OLLAMA_HOST" in str(e) or "NOTES_REPO_URL" in str(e)

        finally:
            # Restore original values
            if original_host:
                os.environ["OLLAMA_HOST"] = original_host
            if original_repo:
                os.environ["NOTES_REPO_URL"] = original_repo

    def test_env_vars_are_read(self):
        """Test that environment variables override defaults."""
        import os

        original_host = os.environ.get("OLLAMA_HOST")
        original_model = os.environ.get("OLLAMA_MODEL")
        original_repo = os.environ.get("NOTES_REPO_URL")
        original_cache = os.environ.get("NOTES_CACHE_DIR")

        try:
            os.environ["OLLAMA_HOST"] = "http://custom:11434"
            os.environ["OLLAMA_MODEL"] = "llama2"
            os.environ["NOTES_REPO_URL"] = "https://token@github.com/custom/repo.git"
            os.environ["NOTES_CACHE_DIR"] = "./custom-cache"

            config = Config()
            assert config.ollama_host == "http://custom:11434"
            assert config.ollama_model == "llama2"
            assert config.notes_repo_url == "https://token@github.com/custom/repo.git"
            assert config.notes_cache_dir == "./custom-cache"

        finally:
            # Restore original values
            if original_host:
                os.environ["OLLAMA_HOST"] = original_host
            elif "OLLAMA_HOST" in os.environ:
                del os.environ["OLLAMA_HOST"]

            if original_model:
                os.environ["OLLAMA_MODEL"] = original_model
            elif "OLLAMA_MODEL" in os.environ:
                del os.environ["OLLAMA_MODEL"]

            if original_repo:
                os.environ["NOTES_REPO_URL"] = original_repo
            elif "NOTES_REPO_URL" in os.environ:
                del os.environ["NOTES_REPO_URL"]

            if original_cache:
                os.environ["NOTES_CACHE_DIR"] = original_cache
            elif "NOTES_CACHE_DIR" in os.environ:
                del os.environ["NOTES_CACHE_DIR"]

    def test_optional_env_vars_have_defaults(self):
        """Test that optional env vars use defaults when not set."""
        import os

        # Set required env vars first
        os.environ["OLLAMA_HOST"] = "http://localhost:11434"
        os.environ["NOTES_REPO_URL"] = "https://test@github.com/test/repo.git"

        # Ensure optional vars are not set
        for var in ["CHUNK_SIZE", "CHUNK_OVERLAP", "TOP_K", "EMBEDDING_MODEL"]:
            os.environ.pop(var, None)

        try:
            config = Config()
            assert config.chunk_size == 1000
            assert config.chunk_overlap == 200
            assert config.top_k == 5
        finally:
            # Clean up
            os.environ.pop("OLLAMA_HOST", None)
            os.environ.pop("NOTES_REPO_URL", None)


class TestDefaultConfig:
    """Test DEFAULT_CONFIG constant."""

    def test_default_config_has_all_fields(self):
        """Test that DEFAULT_CONFIG contains all expected keys."""
        assert "ollama_host" in DEFAULT_CONFIG
        assert "ollama_model" in DEFAULT_CONFIG
        assert "notes_repo_url" in DEFAULT_CONFIG
        assert "notes_cache_dir" in DEFAULT_CONFIG
        assert "chunk_size" in DEFAULT_CONFIG
        assert "chunk_overlap" in DEFAULT_CONFIG
        assert "top_k" in DEFAULT_CONFIG
        assert "embedding_model" in DEFAULT_CONFIG
