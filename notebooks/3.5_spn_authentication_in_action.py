# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 3.5: SPN Authentication in Action
# MAGIC
# MAGIC ## DO NOT RUN — Demonstration/Reference Only
# MAGIC
# MAGIC This notebook demonstrates how to use a Service Principal (SPN)
# MAGIC to connect to Lakebase instead of the user token.
# MAGIC Requires the SPN credentials created in notebook 3.4.
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Loading SPN credentials from Databricks Secret Scopes
# MAGIC - Authenticating to Lakebase with OAuth client credentials
# MAGIC - Using LakebaseMemory with SPN credentials

# COMMAND ----------

import os
from uuid import uuid4

from databricks.sdk import WorkspaceClient
from loguru import logger
from pyspark.sql import SparkSession

from stackoverflow_curator.config import get_env, load_config
from stackoverflow_curator.memory import LakebaseMemory

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load SPN Credentials from Secret Scope

# COMMAND ----------

# DO NOT RUN — requires SPN created in notebook 3.4
scope_name = "hf-agent-scope"
os.environ["DATABRICKS_CLIENT_ID"] = dbutils.secrets.get(scope_name, "client_id")  # noqa: F821
os.environ["DATABRICKS_CLIENT_SECRET"] = dbutils.secrets.get(scope_name, "client_secret")  # noqa: F821

logger.info("SPN credentials loaded from secret scope")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Connect to Lakebase with SPN

# COMMAND ----------

_user_prefix = w.current_user.me().user_name.split("@")[0].replace(".", "-")
project_id = f"{_user_prefix}-lakebase"
logger.info(f"Connecting to Lakebase project: {project_id}")

memory = LakebaseMemory(project_id=project_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Test Session with SPN

# COMMAND ----------

session_id = f"spn-test-session-{uuid4()}"

test_messages = [
    {"role": "user", "content": "How do I use AutoTokenizer?"},
    {
        "role": "assistant",
        "content": "Use AutoTokenizer.from_pretrained('model-name') to load a tokenizer.",
    },
    {"role": "user", "content": "How do I encode a sentence?"},
]

memory.save_messages(session_id, test_messages)
logger.info(f"Saved {len(test_messages)} messages to session: {session_id}")

# COMMAND ----------

loaded_messages = memory.load_messages(session_id)
logger.info(f"Loaded {len(loaded_messages)} messages from session: {session_id}")
for msg in loaded_messages:
    logger.info(f"  {msg['role']}: {msg['content'][:60]}...")

# COMMAND ----------

memory.delete_session(session_id)
logger.info(f"Deleted session: {session_id}")

memory.close()
