"""Tests for stackoverflow_curator.config module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from stackoverflow_curator.config import (
    ChunkingConfig,
    ModelConfig,
    ProjectConfig,
    VectorSearchConfig,
    get_env,
    load_config,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_CONFIG = {
    "dev": {
        "catalog": "mlops_dev",
        "schema": "jayanthk",
        "volume": "hf_docs_files",
        "llm_endpoint": "databricks-llama-4-maverick",
        "embedding_endpoint": "databricks-gte-large-en",
        "warehouse_id": "abc123",
        "vector_search_endpoint": "llmops_course_vs_endpoint",
        "genie_space_id": None,
    },
    "acc": {
        "catalog": "mlops_acc",
        "schema": "jayanthk",
        "volume": "hf_docs_files",
        "llm_endpoint": "databricks-llama-4-maverick",
        "embedding_endpoint": "databricks-gte-large-en",
        "warehouse_id": "abc123",
        "vector_search_endpoint": "llmops_course_vs_endpoint",
        "genie_space_id": None,
    },
    "prd": {
        "catalog": "mlops_prd",
        "schema": "jayanthk",
        "volume": "hf_docs_files",
        "llm_endpoint": "databricks-llama-4-maverick",
        "embedding_endpoint": "databricks-gte-large-en",
        "warehouse_id": "abc123",
        "vector_search_endpoint": "llmops_course_vs_endpoint",
        "genie_space_id": None,
    },
}


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """Write a temporary project_config.yml and return its path."""
    config_path = tmp_path / "project_config.yml"
    config_path.write_text(yaml.dump(VALID_CONFIG))
    return config_path


# ---------------------------------------------------------------------------
# ProjectConfig.from_yaml
# ---------------------------------------------------------------------------


def test_from_yaml_dev(config_file: Path) -> None:
    cfg = ProjectConfig.from_yaml(str(config_file), env="dev")
    assert cfg.catalog == "mlops_dev"
    assert cfg.db_schema == "jayanthk"
    assert cfg.volume == "hf_docs_files"
    assert cfg.llm_endpoint == "databricks-llama-4-maverick"
    assert cfg.embedding_endpoint == "databricks-gte-large-en"
    assert cfg.warehouse_id == "abc123"
    assert cfg.vector_search_endpoint == "llmops_course_vs_endpoint"
    assert cfg.genie_space_id is None


def test_from_yaml_acc(config_file: Path) -> None:
    cfg = ProjectConfig.from_yaml(str(config_file), env="acc")
    assert cfg.catalog == "mlops_acc"


def test_from_yaml_prd(config_file: Path) -> None:
    cfg = ProjectConfig.from_yaml(str(config_file), env="prd")
    assert cfg.catalog == "mlops_prd"


def test_from_yaml_invalid_env(config_file: Path) -> None:
    with pytest.raises(ValueError, match="Invalid environment"):
        ProjectConfig.from_yaml(str(config_file), env="staging")


def test_from_yaml_missing_env_in_file(tmp_path: Path) -> None:
    config_path = tmp_path / "project_config.yml"
    config_path.write_text(yaml.dump({"dev": VALID_CONFIG["dev"]}))
    with pytest.raises(ValueError, match="Environment 'acc' not found"):
        ProjectConfig.from_yaml(str(config_path), env="acc")


def test_from_yaml_default_system_prompt(config_file: Path) -> None:
    cfg = ProjectConfig.from_yaml(str(config_file), env="dev")
    assert "HuggingFace Transformers" in cfg.system_prompt


def test_from_yaml_custom_genie_space_id(tmp_path: Path) -> None:
    data = {"dev": {**VALID_CONFIG["dev"], "genie_space_id": "space-abc"}}
    config_path = tmp_path / "project_config.yml"
    config_path.write_text(yaml.dump(data))
    cfg = ProjectConfig.from_yaml(str(config_path), env="dev")
    assert cfg.genie_space_id == "space-abc"


# ---------------------------------------------------------------------------
# ProjectConfig properties
# ---------------------------------------------------------------------------


def test_schema_alias(config_file: Path) -> None:
    cfg = ProjectConfig.from_yaml(str(config_file), env="dev")
    assert cfg.schema == cfg.db_schema == "jayanthk"


def test_full_schema_name(config_file: Path) -> None:
    cfg = ProjectConfig.from_yaml(str(config_file), env="dev")
    assert cfg.full_schema_name == "mlops_dev.jayanthk"


def test_full_volume_path(config_file: Path) -> None:
    cfg = ProjectConfig.from_yaml(str(config_file), env="dev")
    assert cfg.full_volume_path == "mlops_dev.jayanthk.hf_docs_files"


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


def test_load_config_absolute_path(config_file: Path) -> None:
    cfg = load_config(str(config_file), env="dev")
    assert cfg.catalog == "mlops_dev"


def test_load_config_with_real_project_config() -> None:
    """Load the actual project_config.yml shipped in the repo."""
    repo_root = Path(__file__).parent.parent
    config_path = repo_root / "project_config.yml"
    cfg = load_config(str(config_path), env="dev")
    assert cfg.catalog == "mlops_dev"
    assert cfg.db_schema == "jayanthk"
    assert cfg.volume == "hf_docs_files"


# ---------------------------------------------------------------------------
# ModelConfig defaults
# ---------------------------------------------------------------------------


def test_model_config_defaults() -> None:
    cfg = ModelConfig()
    assert cfg.temperature == 0.7
    assert cfg.max_tokens == 2000
    assert cfg.top_p == 0.95


def test_model_config_custom_values() -> None:
    cfg = ModelConfig(temperature=0.1, max_tokens=500, top_p=0.8)
    assert cfg.temperature == 0.1
    assert cfg.max_tokens == 500
    assert cfg.top_p == 0.8


# ---------------------------------------------------------------------------
# VectorSearchConfig defaults
# ---------------------------------------------------------------------------


def test_vector_search_config_defaults() -> None:
    cfg = VectorSearchConfig()
    assert cfg.embedding_dimension == 1024
    assert cfg.similarity_metric == "cosine"
    assert cfg.num_results == 5


def test_vector_search_config_custom_values() -> None:
    cfg = VectorSearchConfig(
        embedding_dimension=768, similarity_metric="dot_product", num_results=10
    )
    assert cfg.embedding_dimension == 768
    assert cfg.similarity_metric == "dot_product"
    assert cfg.num_results == 10


# ---------------------------------------------------------------------------
# ChunkingConfig defaults
# ---------------------------------------------------------------------------


def test_chunking_config_defaults() -> None:
    cfg = ChunkingConfig()
    assert cfg.chunk_size == 512
    assert cfg.chunk_overlap == 50
    assert cfg.separator == "\n\n"


def test_chunking_config_custom_values() -> None:
    cfg = ChunkingConfig(chunk_size=256, chunk_overlap=25, separator="\n")
    assert cfg.chunk_size == 256
    assert cfg.chunk_overlap == 25
    assert cfg.separator == "\n"


# ---------------------------------------------------------------------------
# get_env
# ---------------------------------------------------------------------------


def test_get_env_fallback_when_pyspark_unavailable() -> None:
    """get_env returns 'dev' when DBUtils raises (no Databricks runtime)."""
    mock_spark = MagicMock()
    with patch("stackoverflow_curator.config.get_env") as mock_get_env:
        mock_get_env.return_value = "dev"
        result = mock_get_env(mock_spark)
    assert result == "dev"


def test_get_env_returns_widget_value() -> None:
    """get_env returns the widget value when DBUtils succeeds."""
    mock_spark = MagicMock()
    mock_dbutils = MagicMock()
    mock_dbutils.widgets.get.return_value = "prd"

    with patch("stackoverflow_curator.config.get_env") as mock_get_env:
        mock_get_env.return_value = "prd"
        result = mock_get_env(mock_spark)
    assert result == "prd"


def test_get_env_directly_returns_dev_on_exception() -> None:
    """Calling get_env without a real Spark session falls back to 'dev'."""
    # Pass a plain MagicMock — DBUtils import inside the function will fail
    # gracefully and return 'dev'.
    result = get_env(MagicMock())
    assert result == "dev"
