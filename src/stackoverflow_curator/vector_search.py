"""Vector search management for HuggingFace Transformers doc chunks."""

from __future__ import annotations

from databricks.vector_search.client import VectorSearchClient
from loguru import logger

from stackoverflow_curator.config import ProjectConfig


class VectorSearchManager:
    """Manages vector search endpoints and indexes for HF doc chunks."""

    def __init__(
        self,
        config: ProjectConfig,
        endpoint_name: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        self.config = config
        self.endpoint_name = endpoint_name or config.vector_search_endpoint
        self.embedding_model = embedding_model or config.embedding_endpoint
        self.catalog = config.catalog
        self.schema = config.schema

        self.client = VectorSearchClient()
        self.index_name = f"{self.catalog}.{self.schema}.hf_docs_index"

    def create_endpoint_if_not_exists(self) -> None:
        """Create the vector search endpoint if it does not already exist."""
        endpoints_response = self.client.list_endpoints()
        endpoints = (
            endpoints_response.get("endpoints", [])
            if isinstance(endpoints_response, dict)
            else []
        )
        endpoint_exists = any(
            (ep.get("name") if isinstance(ep, dict) else getattr(ep, "name", None))
            == self.endpoint_name
            for ep in endpoints
        )

        if not endpoint_exists:
            logger.info(f"Creating vector search endpoint: {self.endpoint_name}")
            self.client.create_endpoint_and_wait(
                name=self.endpoint_name,
                endpoint_type="STANDARD",
            )
            logger.info(f"Endpoint created: {self.endpoint_name}")
        else:
            logger.info(f"Endpoint already exists: {self.endpoint_name}")

    def create_or_get_index(self) -> object:
        """Create or retrieve the delta-sync vector search index."""
        self.create_endpoint_if_not_exists()
        source_table = f"{self.catalog}.{self.schema}.hf_doc_chunks"

        # Try to get existing index
        try:
            index = self.client.get_index(index_name=self.index_name)
            logger.info(f"Index exists: {self.index_name}")
            return index
        except Exception:
            logger.info(f"Index {self.index_name} not found, will create it")

        # Create delta-sync index
        try:
            index = self.client.create_delta_sync_index(
                endpoint_name=self.endpoint_name,
                source_table_name=source_table,
                index_name=self.index_name,
                pipeline_type="TRIGGERED",
                primary_key="id",
                embedding_source_column="text",
                embedding_model_endpoint_name=self.embedding_model,
            )
            logger.info(f"Index created: {self.index_name}")
            return index
        except Exception as e:
            if "RESOURCE_ALREADY_EXISTS" not in str(e):
                raise
            logger.info(f"Index already exists: {self.index_name}")
            return self.client.get_index(index_name=self.index_name)

    def sync_index(self) -> None:
        """Trigger a manual sync of the vector search index."""
        index = self.create_or_get_index()
        logger.info(f"Syncing index: {self.index_name}")
        index.sync()
        logger.info("Index sync triggered")

    def search(
        self,
        query: str,
        num_results: int = 5,
        filters: dict | None = None,
    ) -> dict:
        """Run a similarity search against the index."""
        index = self.client.get_index(index_name=self.index_name)
        return index.similarity_search(
            query_text=query,
            columns=[
                "id",
                "text",
                "title",
                "section_title",
                "primary_category",
            ],
            num_results=num_results,
            filters=filters,
        )
