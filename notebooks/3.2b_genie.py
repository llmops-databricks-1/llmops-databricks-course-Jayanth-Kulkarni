# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 3.2b: Genie Space Integration
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Creating a personal Genie space over HuggingFace doc tables
# MAGIC - Configuring data sources and column hints
# MAGIC - Starting conversations with Genie
# MAGIC - Using Genie for natural language queries over doc metadata
# MAGIC
# MAGIC **What is Genie?**
# MAGIC - Databricks Genie is an AI-powered data analyst
# MAGIC - Converts natural language questions to SQL queries
# MAGIC - Executes queries and returns results
# MAGIC - Can be integrated with agents via MCP

# COMMAND ----------

import json

from databricks.sdk import WorkspaceClient
from loguru import logger
from pyspark.sql import SparkSession

from stackoverflow_curator.config import get_env, load_config

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

env = get_env(spark)
cfg = load_config("../project_config.yml", env)

catalog = cfg.catalog
schema = cfg.schema

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Check for Existing Genie Space
# MAGIC
# MAGIC First, check if we already have a Genie space configured.

# COMMAND ----------

w = WorkspaceClient()

_user_prefix = w.current_user.me().user_name.split("@")[0].replace(".", "_")
_warehouse_name = f"{_user_prefix}_hf_docs_warehouse"
_space_title = f"{_user_prefix}-hf-docs-curator-space"

logger.info(f"Personal warehouse name: {_warehouse_name}")
logger.info(f"Personal Genie space title: {_space_title}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. SQL Warehouse
# MAGIC
# MAGIC Use the shared warehouse from config (user doesn't have permission to create new ones).

# COMMAND ----------

warehouse_id = cfg.warehouse_id
logger.info(f"Using shared warehouse from config: {warehouse_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Configure Genie Space
# MAGIC
# MAGIC Create a personal Genie space pointing at `hf_doc_sources` and `hf_doc_chunks`.

# COMMAND ----------

serialized_space = {
    "version": 1,
    "data_sources": {
        "tables": [
            {
                "identifier": f"{catalog}.{schema}.hf_doc_chunks",
                "column_configs": [
                    {"column_name": "chunk_index"},
                    {"column_name": "doc_id", "get_example_values": True},
                    {"column_name": "id", "get_example_values": True},
                    {
                        "column_name": "primary_category",
                        "get_example_values": True,
                        "build_value_dictionary": True,
                    },
                    {
                        "column_name": "section_title",
                        "get_example_values": True,
                        "build_value_dictionary": True,
                    },
                    {"column_name": "text"},
                    {
                        "column_name": "title",
                        "get_example_values": True,
                        "build_value_dictionary": True,
                    },
                ],
            },
            {
                "identifier": f"{catalog}.{schema}.hf_doc_sources",
                "column_configs": [
                    {"column_name": "doc_id", "get_example_values": True},
                    {"column_name": "ingest_ts"},
                    {
                        "column_name": "primary_category",
                        "get_example_values": True,
                        "build_value_dictionary": True,
                    },
                    {
                        "column_name": "title",
                        "get_example_values": True,
                        "build_value_dictionary": True,
                    },
                    {"column_name": "url", "get_example_values": True},
                ],
            },
        ]
    },
}

# Delete existing space if present so it is always recreated with the current warehouse
existing_spaces = {s.title: s for s in (w.genie.list_spaces().spaces or [])}

if _space_title in existing_spaces:
    old_space_id = existing_spaces[_space_title].space_id
    try:
        w.genie.delete_space(space_id=old_space_id)
        logger.info(
            f"Deleted existing Genie Space {old_space_id} to recreate with current warehouse"
        )
    except Exception as e:
        logger.warning(f"Could not delete existing space: {type(e).__name__}: {e}")

space = w.genie.create_space(
    warehouse_id=warehouse_id,
    serialized_space=json.dumps(serialized_space),
    title=_space_title,
)
space_id = space.space_id
logger.info(f"Created Genie Space: {space_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Verify Genie Space

# COMMAND ----------

try:
    space = w.genie.get_space(space_id=space_id, include_serialized_space=True)
    logger.info(f"Genie Space ID: {space_id}")
    if space.serialized_space:
        logger.info(f"Space config: {json.loads(space.serialized_space)}")
except Exception as e:
    logger.warning(
        f"Could not get space details (may need 'Can Edit' permission): {type(e).__name__}"
    )
    logger.info(f"Genie Space ID: {space_id} — proceeding with conversations anyway")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Start a Conversation
# MAGIC
# MAGIC Ask Genie a natural language question about the HuggingFace doc metadata.

# COMMAND ----------

try:
    conversation = w.genie.start_conversation_and_wait(
        space_id=space_id,
        content="How many doc chunks are there per category?",
    )
    logger.info(f"Conversation started: {conversation.conversation_id}")
    logger.info(conversation.as_dict())
except Exception as e:
    logger.warning(f"Genie conversation failed: {type(e).__name__}: {e}")
    conversation = None

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Continue the Conversation

# COMMAND ----------

if conversation:
    try:
        message = w.genie.create_message_and_wait(
            space_id=space_id,
            conversation_id=conversation.conversation_id,
            content="Which documents have the most chunks? Show top 10.",
        )
        logger.info(message.as_dict())
    except Exception as e:
        logger.warning(f"Genie follow-up message failed: {type(e).__name__}: {e}")
else:
    logger.info("Skipping follow-up — no active conversation.")
