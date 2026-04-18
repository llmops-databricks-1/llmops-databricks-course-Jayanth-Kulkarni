"""Tests for stackoverflow_curator.vector_search module.

Only tests the import and class instantiation contract — the actual
Databricks VectorSearchClient is not available in CI.
"""

from unittest.mock import MagicMock, patch

from stackoverflow_curator.config import ProjectConfig


def _make_config():
    return ProjectConfig(
        catalog="mlops_dev",
        schema="jayanthk",
        volume="hf_docs_files",
        llm_endpoint="databricks-llama-4-maverick",
        embedding_endpoint="databricks-gte-large-en",
        warehouse_id="abc123",
        vector_search_endpoint="llmops_course_vs_endpoint",
    )


@patch(
    "stackoverflow_curator.vector_search.VectorSearchClient",
    return_value=MagicMock(),
)
def test_manager_init_defaults(mock_client):
    from stackoverflow_curator.vector_search import VectorSearchManager

    cfg = _make_config()
    mgr = VectorSearchManager(config=cfg)

    assert mgr.endpoint_name == "llmops_course_vs_endpoint"
    assert mgr.embedding_model == "databricks-gte-large-en"
    assert mgr.index_name == "mlops_dev.jayanthk.hf_docs_index"
    assert mgr.catalog == "mlops_dev"
    assert mgr.schema == "jayanthk"


@patch(
    "stackoverflow_curator.vector_search.VectorSearchClient",
    return_value=MagicMock(),
)
def test_manager_custom_endpoint(mock_client):
    from stackoverflow_curator.vector_search import VectorSearchManager

    cfg = _make_config()
    mgr = VectorSearchManager(
        config=cfg,
        endpoint_name="custom_ep",
        embedding_model="custom_model",
    )

    assert mgr.endpoint_name == "custom_ep"
    assert mgr.embedding_model == "custom_model"


@patch(
    "stackoverflow_curator.vector_search.VectorSearchClient",
    return_value=MagicMock(),
)
def test_index_name_format(mock_client):
    from stackoverflow_curator.vector_search import VectorSearchManager

    cfg = _make_config()
    mgr = VectorSearchManager(config=cfg)

    assert mgr.index_name == f"{cfg.catalog}.{cfg.schema}.hf_docs_index"
