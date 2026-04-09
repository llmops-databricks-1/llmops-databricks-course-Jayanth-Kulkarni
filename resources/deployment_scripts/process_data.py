# Databricks notebook source
# MAGIC %md
# MAGIC # Data Processing Pipeline
# MAGIC
# MAGIC This notebook processes HuggingFace Transformers docs and syncs the vector
# MAGIC search index. Runs on a schedule to keep the knowledge base up to date.
# MAGIC
# MAGIC Pipeline steps:
# MAGIC 1. Chunk raw markdown docs into embedding-friendly pieces
# MAGIC 2. Write chunks to the hf_doc_chunks Delta table
# MAGIC 3. Sync the vector search index

# COMMAND ----------

from loguru import logger
from pyspark.sql import SparkSession

from stackoverflow_curator.config import get_env, load_config
from stackoverflow_curator.data_processor import DataProcessor
from stackoverflow_curator.vector_search import VectorSearchManager

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

env = get_env(spark)
cfg = load_config("../../project_config.yml", env=env)

logger.info("Configuration loaded:")
logger.info(f"  Environment: {env}")
logger.info(f"  Catalog: {cfg.catalog}")
logger.info(f"  Schema: {cfg.schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Process and Chunk Documents

# COMMAND ----------

processor = DataProcessor(spark=spark, config=cfg)
processor.process_and_save()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Sync Vector Search Index

# COMMAND ----------

vs_manager = VectorSearchManager(config=cfg)
vs_manager.sync_index()

logger.info("Data processing pipeline complete!")
