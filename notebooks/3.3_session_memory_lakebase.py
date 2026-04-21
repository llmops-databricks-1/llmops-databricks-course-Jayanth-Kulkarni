# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 3.3: Session Memory with Lakebase
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Lakebase (Databricks PostgreSQL) for session persistence
# MAGIC - PostgresAPI: project/branch/endpoint model
# MAGIC - Managing conversation history with JSONB
# MAGIC - Building stateful HuggingFace Docs agents with LakebaseMemory
# MAGIC
# MAGIC > **Note**: Lakebase must be enabled for your workspace by an admin.
# MAGIC > If it is not enabled, sections 1–4 will be skipped and the notebook
# MAGIC > will log a warning.

# COMMAND ----------

import json
import os
import urllib.parse
from uuid import uuid4

os.environ.setdefault("PSYCOPG_IMPL", "python")

import psycopg
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.postgres import (
    PostgresAPI,
    Project,
    ProjectDefaultEndpointSettings,
    ProjectSpec,
)
from google.protobuf.duration_pb2 import Duration
from loguru import logger
from openai import OpenAI
from pyspark.sql import SparkSession

from stackoverflow_curator.config import get_env, load_config
from stackoverflow_curator.memory import LakebaseMemory

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

w = WorkspaceClient()
pg_api = PostgresAPI(w.api_client)

_user_prefix = w.current_user.me().user_name.split("@")[0].replace(".", "-")
project_id = f"{_user_prefix}-lakebase"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Create Personal Lakebase Project
# MAGIC
# MAGIC **Lakebase** is Databricks' managed PostgreSQL service:
# MAGIC - Fully managed, scales to 0 when idle
# MAGIC - Integrated with Databricks authentication
# MAGIC - Project → Branch → Endpoint hierarchy
# MAGIC - Ideal for session state, caching, and metadata

# COMMAND ----------

_lakebase_available = False
project = None
conn_string = None

try:
    try:
        project = pg_api.get_project(name=f"projects/{project_id}")
        logger.info(f"Using existing Lakebase project: {project_id}")
    except Exception:
        logger.info(f"Creating new Lakebase project: {project_id}")
        project = pg_api.create_project(
            project_id=project_id,
            project=Project(
                spec=ProjectSpec(
                    display_name=project_id,
                    default_endpoint_settings=ProjectDefaultEndpointSettings(
                        autoscaling_limit_min_cu=1,
                        autoscaling_limit_max_cu=4,
                        suspend_timeout_duration=Duration(seconds=300),
                    ),
                ),
            ),
        ).wait()
        logger.info(f"Created Lakebase project: {project_id}")
    _lakebase_available = True
except Exception as e:
    logger.warning(f"Lakebase not available: {type(e).__name__}: {e}")
    logger.warning("Skipping Lakebase sections — ask your workspace admin to enable Lakebase.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Connect and Create Table

# COMMAND ----------

if _lakebase_available:
    default_branch = next(iter(pg_api.list_branches(parent=project.name)))
    endpoint = next(iter(pg_api.list_endpoints(parent=default_branch.name)))
    host = endpoint.status.hosts.host

    pg_credential = pg_api.generate_database_credential(endpoint=endpoint.name)
    user = w.current_user.me()
    username = urllib.parse.quote_plus(user.user_name)
    conn_string = (
        f"postgresql://{username}:{pg_credential.token}@{host}:5432/"
        "databricks_postgres?sslmode=require"
    )
    logger.info(f"Lakebase host: {host}")
else:
    logger.warning("Skipping: Lakebase not available.")

# COMMAND ----------

if _lakebase_available and conn_string:
    with psycopg.connect(conn_string) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_messages (
                id SERIAL PRIMARY KEY,
                session_id TEXT NOT NULL,
                message_data JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_messages_session_id
            ON session_messages(session_id)
        """)
    logger.info("session_messages table ready")
else:
    logger.warning("Skipping: Lakebase not available.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Save and Load Messages

# COMMAND ----------

if _lakebase_available and conn_string:
    test_session_id = f"test-session-{uuid4()}"
    test_messages = [
        {"role": "user", "content": "How do I load a pretrained model?"},
        {
            "role": "assistant",
            "content": "Use AutoModel.from_pretrained('model-name') to load a model.",
        },
        {"role": "user", "content": "What about tokenizers?"},
    ]

    with psycopg.connect(conn_string) as conn:
        for msg in test_messages:
            conn.execute(
                "INSERT INTO session_messages (session_id, message_data) VALUES (%s, %s)",
                (test_session_id, json.dumps(msg)),
            )
    logger.info(f"Saved {len(test_messages)} messages to session: {test_session_id}")
else:
    logger.warning("Skipping: Lakebase not available.")

# COMMAND ----------

if _lakebase_available and conn_string:
    with psycopg.connect(conn_string) as conn:
        result = conn.execute(
            """
            SELECT message_data, created_at FROM session_messages
            WHERE session_id = %s
            ORDER BY created_at ASC
            """,
            (test_session_id,),
        ).fetchall()

    logger.info(f"Loaded {len(result)} messages:")
    for row in result:
        logger.info(f"  [{row[1]}] {row[0]}")
else:
    logger.warning("Skipping: Lakebase not available.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Using LakebaseMemory Class

# COMMAND ----------

memory = LakebaseMemory(project_id=project_id)

# COMMAND ----------

if memory._available:
    session_id = f"memory-test-{uuid4()}"
    messages = [
        {"role": "user", "content": "What is the pipeline API?"},
        {
            "role": "assistant",
            "content": "The pipeline API is the easiest way to use pretrained models for inference.",
        },
    ]
    memory.save_messages(session_id, messages)
    logger.info(f"Saved messages to session: {session_id}")

    loaded = memory.load_messages(session_id)
    logger.info(f"Loaded {len(loaded)} messages:")
    for msg in loaded:
        logger.info(f"  {msg['role']}: {msg['content'][:60]}...")
else:
    logger.warning("LakebaseMemory unavailable — skipping save/load demo.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Stateful Multi-Turn Conversation with LLM

# COMMAND ----------

_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()  # noqa: F821
client = OpenAI(api_key=_token, base_url=f"{w.config.host}/serving-endpoints")


def chat_with_memory(
    session_id: str, user_message: str, memory: LakebaseMemory
) -> str:
    """Chat with LLM using session memory for context."""
    previous_messages = memory.load_messages(session_id)
    messages = (
        [
            {
                "role": "system",
                "content": (
                    "You are a helpful HuggingFace Transformers assistant. "
                    "Answer questions about the HuggingFace Transformers library."
                ),
            }
        ]
        + previous_messages
        + [{"role": "user", "content": user_message}]
    )
    response = client.chat.completions.create(
        model=cfg.llm_endpoint,
        messages=messages,
    )
    assistant_response = response.choices[0].message.content
    memory.save_messages(
        session_id,
        [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_response},
        ],
    )
    return assistant_response


logger.info("Chat function with memory created")

# COMMAND ----------

if memory._available:
    agent_session_id = f"hf-docs-session-{uuid4()}"

    response1 = chat_with_memory(
        agent_session_id, "What is the pipeline API in HuggingFace Transformers?", memory
    )
    logger.info(f"Response 1: {response1[:200]}...")

    response2 = chat_with_memory(
        agent_session_id, "What tasks does it support?", memory
    )
    logger.info(f"Response 2: {response2[:200]}...")

    full_conversation = memory.load_messages(agent_session_id)
    logger.info(f"Full conversation ({len(full_conversation)} messages):")
    for i, msg in enumerate(full_conversation, 1):
        content = (
            msg["content"][:100] + "..."
            if len(msg["content"]) > 100
            else msg["content"]
        )
        logger.info(f"  {i}. [{msg['role']}] {content}")
else:
    logger.warning("LakebaseMemory unavailable — skipping multi-turn demo.")
    logger.warning("To enable: ask your workspace admin to enable Lakebase for workspace 3158303576563862.")

memory.close()
