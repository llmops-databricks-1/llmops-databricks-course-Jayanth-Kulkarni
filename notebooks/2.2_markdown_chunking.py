# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 2.2: Markdown Parsing & Chunking
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Processing raw markdown from HuggingFace docs (ingested in Week 1)
# MAGIC - Heading-aware chunking strategy
# MAGIC - Text cleaning for embeddings
# MAGIC - Writing chunks to a Silver Delta table
# MAGIC
# MAGIC ### Our Approach
# MAGIC
# MAGIC Unlike the reference project which uses Databricks AI Parse Documents for PDFs,
# MAGIC our data is already **raw Markdown** fetched from GitHub in Week 1.
# MAGIC We use a heading-aware chunking strategy that splits on `##` boundaries,
# MAGIC preserving the document's natural semantic structure.

# COMMAND ----------

from loguru import logger
from pyspark.sql import SparkSession

from stackoverflow_curator.config import get_env, load_config
from stackoverflow_curator.data_processor import DataProcessor

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

env = get_env(spark)
cfg = load_config("../project_config.yml", env)

logger.info(f"Catalog: {cfg.catalog}, Schema: {cfg.schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Parsing Tools Comparison
# MAGIC
# MAGIC | Tool | Pros | Cons | Best For |
# MAGIC |------|------|------|----------|
# MAGIC | **AI Parse Documents** | AI-powered, handles complex layouts | Databricks-specific, cost per page | Complex PDFs, tables |
# MAGIC | **PyPDF2** | Simple, free, pure Python | Poor with complex layouts | Simple text extraction |
# MAGIC | **Heading-aware split** | Preserves doc structure, fast | Only works for markdown | Markdown documentation |
# MAGIC | **Unstructured.io** | ML-powered, good chunking | External service, API costs | Modern RAG pipelines |
# MAGIC
# MAGIC **We use heading-aware splitting** because our source data is already structured
# MAGIC Markdown from the HuggingFace Transformers repo.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Inspect Source Data from Week 1
# MAGIC
# MAGIC The `hf_doc_sources` table was populated in Week 1 by fetching raw markdown
# MAGIC from `docs/source/en/*.md` in the transformers GitHub repo.

# COMMAND ----------

sources_table = f"{cfg.catalog}.{cfg.schema}.hf_doc_sources"
sources_df = spark.table(sources_table)

total = sources_df.count()
fetched = sources_df.filter("is_fetched = true").count()
logger.info(f"Source documents: {fetched}/{total} fetched")

sources_df.select("id", "title", "primary_category", "is_fetched").show(10, truncate=50)

# COMMAND ----------

# Preview a single document's raw content
sample = (
    sources_df.filter("is_fetched = true").select("id", "title", "raw_content").first()
)
logger.info(f"Sample doc: {sample['title']} ({sample['id']})")
logger.info(f"Raw content length: {len(sample['raw_content']):,} chars")
logger.info(f"\nFirst 500 chars:\n{sample['raw_content'][:500]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Process and Chunk All Documents
# MAGIC
# MAGIC The `DataProcessor` class handles the full pipeline:
# MAGIC 1. Read all fetched docs from `hf_doc_sources`
# MAGIC 2. Split each document by `##` headings
# MAGIC 3. Clean markdown artefacts (images, HTML, links)
# MAGIC 4. Apply fixed-size splitting for oversized sections
# MAGIC 5. Write chunks to `hf_doc_chunks` Delta table
# MAGIC 6. Enable Change Data Feed for vector search sync

# COMMAND ----------

processor = DataProcessor(spark=spark, config=cfg)
processor.process_and_save()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Inspect the Chunks Table

# COMMAND ----------

chunks_table = f"{cfg.catalog}.{cfg.schema}.hf_doc_chunks"
chunks_df = spark.table(chunks_table)

logger.info(f"Total chunks: {chunks_df.count()}")
chunks_df.select("id", "doc_id", "section_title", "title", "primary_category").show(
    10, truncate=50
)

# COMMAND ----------

# Preview a chunk's content
sample_chunk = chunks_df.select("id", "section_title", "text").first()
logger.info(f"Chunk ID: {sample_chunk['id']}")
logger.info(f"Section: {sample_chunk['section_title']}")
logger.info(f"Text ({len(sample_chunk['text'])} chars):\n{sample_chunk['text'][:500]}")

# COMMAND ----------
