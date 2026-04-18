# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 2.4: Embeddings & Vector Search
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Understanding embeddings
# MAGIC - Different embedding models
# MAGIC - Creating vector search endpoints
# MAGIC - Creating and syncing vector search indexes
# MAGIC - Querying with similarity search
# MAGIC - Advanced options: filters, hybrid search

# COMMAND ----------

from loguru import logger
from pyspark.sql import SparkSession

from stackoverflow_curator.config import get_env, load_config
from stackoverflow_curator.vector_search import VectorSearchManager

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

env = get_env(spark)
cfg = load_config("../project_config.yml", env)
catalog = cfg.catalog
schema = cfg.schema

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Understanding Embeddings
# MAGIC
# MAGIC **Embeddings** are numerical representations of text that capture semantic meaning.
# MAGIC
# MAGIC ### Key Concepts:
# MAGIC
# MAGIC - **Vector**: Array of numbers (e.g., [0.1, -0.3, 0.5, ...])
# MAGIC - **Dimension**: Length of the vector (e.g., 384, 768, 1024)
# MAGIC - **Semantic Similarity**: Similar meanings = similar vectors
# MAGIC - **Distance Metrics**: Cosine similarity, Euclidean distance, dot product
# MAGIC
# MAGIC ### How it Works:
# MAGIC
# MAGIC ```
# MAGIC Text: "fine-tune a model"
# MAGIC   -> (Embedding Model)
# MAGIC Vector: [0.23, -0.15, 0.67, ..., 0.42]  # 1024 dimensions
# MAGIC
# MAGIC Text: "train a neural network"
# MAGIC   -> (Embedding Model)
# MAGIC Vector: [0.25, -0.13, 0.65, ..., 0.40]  # Similar to above!
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Embedding Models Comparison
# MAGIC
# MAGIC | Model | Dimensions | Max Tokens | Best For |
# MAGIC |-------|-----------|------------|----------|
# MAGIC | **databricks-bge-large-en** | 1024 | 512 | General purpose, high quality |
# MAGIC | **databricks-gte-large-en** | 1024 | 512 | General purpose, fast |
# MAGIC | **text-embedding-ada-002** (OpenAI) | 1536 | 8191 | High quality, expensive |
# MAGIC | **e5-large-v2** | 1024 | 512 | Open source, good quality |
# MAGIC | **all-MiniLM-L6-v2** | 384 | 512 | Fast, smaller, lower quality |
# MAGIC
# MAGIC **For this project, we use `databricks-gte-large-en`** — fast, high-quality,
# MAGIC and free on Databricks.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Vector Search Architecture
# MAGIC
# MAGIC ```
# MAGIC +------------------------------------------+
# MAGIC |     Delta Table (hf_doc_chunks)           |
# MAGIC |  - id                                     |
# MAGIC |  - text                                   |
# MAGIC |  - title, section_title, category         |
# MAGIC +--------------------+---------------------+
# MAGIC                      |
# MAGIC                      | (Automatic delta sync)
# MAGIC                      v
# MAGIC +------------------------------------------+
# MAGIC |     Vector Search Index                   |
# MAGIC |  - Embeddings generated automatically     |
# MAGIC |  - Stored in optimized format             |
# MAGIC |  - Supports similarity search             |
# MAGIC +--------------------+---------------------+
# MAGIC                      |
# MAGIC                      | (Query)
# MAGIC                      v
# MAGIC +------------------------------------------+
# MAGIC |     Search Results                        |
# MAGIC |  - Most similar chunks                    |
# MAGIC |  - With similarity scores                 |
# MAGIC +------------------------------------------+
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create Vector Search Endpoint

# COMMAND ----------

vs_manager = VectorSearchManager(
    config=cfg,
    endpoint_name=cfg.vector_search_endpoint,
    embedding_model=cfg.embedding_endpoint,
)

logger.info(f"Vector Search Endpoint: {vs_manager.endpoint_name}")
logger.info(f"Embedding Model: {vs_manager.embedding_model}")
logger.info(f"Index Name: {vs_manager.index_name}")

# COMMAND ----------

vs_manager.create_endpoint_if_not_exists()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Endpoint Types:
# MAGIC
# MAGIC - **STANDARD**: General purpose, good performance
# MAGIC - **STANDARD_LARGE**: Higher throughput, more expensive
# MAGIC
# MAGIC For development and most production workloads, STANDARD is sufficient.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Create Vector Search Index

# COMMAND ----------

# Creates the index if it doesn't exist, configures it with the embedding model,
# and sets up delta sync with the hf_doc_chunks table.

index = vs_manager.create_or_get_index()

logger.info("Vector search setup complete!")
logger.info(f"  Index: {vs_manager.index_name}")
logger.info(f"  Source: {vs_manager.catalog}.{vs_manager.schema}.hf_doc_chunks")
logger.info(f"  Embedding Model: {vs_manager.embedding_model}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Index Configuration Options:
# MAGIC
# MAGIC - **pipeline_type**:
# MAGIC   - `TRIGGERED`: Manual sync, good for batch processing
# MAGIC   - `CONTINUOUS`: Auto-sync with Change Data Feed, real-time updates
# MAGIC
# MAGIC - **primary_key**: Unique identifier for each document
# MAGIC
# MAGIC - **embedding_source_column**: The text column to embed
# MAGIC
# MAGIC - **embedding_model_endpoint_name**: Which embedding model to use

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Helper Function for Parsing Results

# COMMAND ----------


def parse_vector_search_results(results):
    """Parse vector search results from array format to dict format."""
    columns = [c["name"] for c in results.get("manifest", {}).get("columns", [])]
    data_array = results.get("result", {}).get("data_array", [])
    return [dict(zip(columns, row_data, strict=False)) for row_data in data_array]


# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Semantic Search with Similarity
# MAGIC
# MAGIC ### How Semantic Search Works
# MAGIC
# MAGIC 1. **Query Embedding**: Convert your search query to a vector
# MAGIC 2. **Similarity Calculation**: Compare query vector to all doc vectors
# MAGIC 3. **Ranking**: Return documents with highest similarity scores
# MAGIC
# MAGIC ### Cosine Similarity
# MAGIC
# MAGIC Measures the angle between two vectors (range: -1 to 1):
# MAGIC - **1.0**: Identical meaning
# MAGIC - **0.8-0.9**: Very similar
# MAGIC - **0.5-0.7**: Somewhat related
# MAGIC - **< 0.5**: Less relevant

# COMMAND ----------

query = "How do I fine-tune a pretrained model with LoRA?"

results = index.similarity_search(
    query_text=query,
    columns=["text", "id", "title", "section_title"],
    num_results=5,
)

logger.info(f"Query: {query}\n")
logger.info("Top 5 Results:")
logger.info("=" * 80)

for i, row in enumerate(parse_vector_search_results(results), 1):
    logger.info(f"\n{i}. Doc: {row.get('title', 'N/A')}")
    logger.info(f"   Section: {row.get('section_title', 'N/A')}")
    logger.info(f"   Chunk ID: {row.get('id', 'N/A')}")
    logger.info(f"   Text preview: {row.get('text', '')[:200]}...")
    logger.info(f"   Score: {row.get('score', 'N/A'):.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Advanced Search: Filters

# COMMAND ----------

# Search with metadata filters — only chunks from "Get started" docs
query = "What is a pipeline?"

results = index.similarity_search(
    query_text=query,
    columns=["text", "id", "title", "primary_category"],
    filters={"primary_category": "Get started"},
    num_results=3,
)

logger.info(f"Query: {query}")
logger.info("Filter: primary_category = 'Get started'\n")
logger.info("Results:")
logger.info("=" * 80)

for i, row in enumerate(parse_vector_search_results(results), 1):
    logger.info(f"\n{i}. {row.get('title', 'N/A')}")
    logger.info(f"   Category: {row.get('primary_category', 'N/A')}")
    logger.info(f"   Text: {row.get('text', '')[:150]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Hybrid Search: Semantic + Keyword
# MAGIC
# MAGIC ### Why Hybrid Search?
# MAGIC
# MAGIC **Semantic search alone** may miss:
# MAGIC - Exact class names (e.g., "AutoModelForCausalLM")
# MAGIC - Method names and API signatures
# MAGIC - Specific configuration parameter names
# MAGIC
# MAGIC **Hybrid search** combines:
# MAGIC - **Semantic search** (embeddings) -> Captures meaning, synonyms
# MAGIC - **Keyword search** (BM25) -> Exact term matching
# MAGIC
# MAGIC ### BM25 (Best Match 25)
# MAGIC
# MAGIC Keyword scoring algorithm that considers:
# MAGIC - **Term frequency**: How often does the term appear?
# MAGIC - **Document length**: Normalize by doc length
# MAGIC - **Inverse document frequency**: Rare terms = higher weight

# COMMAND ----------

query = "AutoModelForSequenceClassification from_pretrained"

results = index.similarity_search(
    query_text=query,
    columns=["text", "id", "title"],
    num_results=5,
    query_type="hybrid",
)

logger.info(f"Query: {query}")
logger.info("Search Type: Hybrid (Semantic + Keyword)\n")
logger.info("Results:")
logger.info("=" * 80)

for i, row in enumerate(parse_vector_search_results(results), 1):
    logger.info(f"\n{i}. {row.get('title', 'N/A')}")
    logger.info(f"   Text: {row.get('text', '')[:200]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Reranking for Higher Precision
# MAGIC
# MAGIC ### The Two-Stage Retrieval Pattern
# MAGIC
# MAGIC **Stage 1: Fast Retrieval** (Bi-encoder)
# MAGIC - Retrieve top 20-50 candidates quickly
# MAGIC - Uses pre-computed embeddings
# MAGIC
# MAGIC **Stage 2: Precise Reranking** (Cross-encoder)
# MAGIC - Score each candidate against the query
# MAGIC - More accurate relevance scoring
# MAGIC - Slower, but only runs on candidates
# MAGIC
# MAGIC | Aspect | Bi-encoder | Cross-encoder |
# MAGIC |--------|-----------|---------------|
# MAGIC | **Speed** | Very fast | Slower |
# MAGIC | **Accuracy** | Good | Excellent |
# MAGIC | **Use case** | Initial retrieval | Reranking |
# MAGIC
# MAGIC > **Note**: `DatabricksReranker` is not yet enabled in this workspace.
# MAGIC > The hybrid search above already provides strong relevance improvements.
# MAGIC > Reranking can be added once it is available.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Search Quality Comparison

# COMMAND ----------

query = "tokenizer encode decode special tokens"

logger.info(f"Query: {query}\n")

# Strategy 1: Basic semantic search
results_basic = index.similarity_search(
    query_text=query,
    columns=["text", "title"],
    num_results=3,
)

logger.info("Strategy 1: Basic Semantic Search")
logger.info("-" * 80)
for i, row in enumerate(parse_vector_search_results(results_basic), 1):
    logger.info(f"{i}. {row.get('title', 'N/A')[:60]}...")

# Strategy 2: Hybrid search
results_hybrid = index.similarity_search(
    query_text=query,
    columns=["text", "title"],
    num_results=3,
    query_type="hybrid",
)

logger.info("\nStrategy 2: Hybrid Search")
logger.info("-" * 80)
for i, row in enumerate(parse_vector_search_results(results_hybrid), 1):
    logger.info(f"{i}. {row.get('title', 'N/A')[:60]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Monitoring and Maintenance

# COMMAND ----------

# Check index status
index_info = vs_manager.client.get_index(
    endpoint_name=vs_manager.endpoint_name,
    index_name=vs_manager.index_name,
)

logger.info("Index Information:")
logger.info(f"  Name: {index_info.name}")
logger.info(f"  Endpoint: {index_info.endpoint_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Index Maintenance:
# MAGIC
# MAGIC ```python
# MAGIC # Sync index manually (for TRIGGERED pipeline)
# MAGIC index.sync()
# MAGIC
# MAGIC # Delete index (if needed)
# MAGIC # vs_manager.client.delete_index(index_name=vs_manager.index_name)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC In this notebook, we learned:
# MAGIC
# MAGIC 1. Understanding embeddings and vector representations
# MAGIC 2. Comparing different embedding models
# MAGIC 3. Creating vector search endpoints
# MAGIC 4. Creating and syncing vector search indexes
# MAGIC 5. Basic similarity search on HuggingFace Transformers docs
# MAGIC 6. Advanced features: filters and hybrid search
# MAGIC 7. Comparing search strategies (semantic vs hybrid)
# MAGIC 8. Best practices and monitoring
