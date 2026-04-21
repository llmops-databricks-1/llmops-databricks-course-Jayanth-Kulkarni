# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 3.1b: Simple RAG with Vector Search
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - What is RAG (Retrieval-Augmented Generation)?
# MAGIC - Using Vector Search for document retrieval
# MAGIC - Enriching prompts with retrieved context
# MAGIC - Building a simple HuggingFace Docs Q&A system
# MAGIC
# MAGIC **RAG Flow:**
# MAGIC ```
# MAGIC User Question
# MAGIC     ↓
# MAGIC Vector Search (retrieve relevant HF doc chunks)
# MAGIC     ↓
# MAGIC Build Prompt (question + context)
# MAGIC     ↓
# MAGIC LLM (generate answer)
# MAGIC     ↓
# MAGIC Response
# MAGIC ```

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
from loguru import logger
from openai import OpenAI
from pyspark.sql import SparkSession

from stackoverflow_curator.config import get_env, load_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

w = WorkspaceClient()
_token = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
)  # noqa: F821

client = OpenAI(api_key=_token, base_url=f"{w.config.host}/serving-endpoints")

vsc = VectorSearchClient(
    workspace_url=w.config.host,
    personal_access_token=_token,
)

logger.info(f"Connected to workspace: {w.config.host}")
logger.info(f"Using LLM endpoint: {cfg.llm_endpoint}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Vector Search Retrieval
# MAGIC
# MAGIC Retrieve relevant HuggingFace Transformers documentation chunks.

# COMMAND ----------


def retrieve_documents(query: str, num_results: int = 5) -> list[dict]:
    """Retrieve relevant HF doc chunks from vector search.

    Args:
        query: The search query
        num_results: Number of documents to retrieve

    Returns:
        List of document dictionaries with title, section, text, and category
    """
    index_name = f"{cfg.catalog}.{cfg.schema}.hf_docs_index"
    index = vsc.get_index(index_name=index_name)

    results = index.similarity_search(
        query_text=query,
        columns=["text", "title", "section_title", "primary_category", "id"],
        num_results=num_results,
        query_type="hybrid",
    )

    documents = []
    if results and "result" in results:
        data_array = results["result"].get("data_array", [])
        for row in data_array:
            documents.append(
                {
                    "text": row[0],
                    "title": row[1],
                    "section_title": row[2],
                    "primary_category": row[3],
                    "chunk_id": row[4],
                }
            )

    return documents


# COMMAND ----------

query = "How do I load a pretrained transformer model?"
docs = retrieve_documents(query, num_results=3)

logger.info(f"Retrieved {len(docs)} doc chunks for query: '{query}'")
for i, doc in enumerate(docs, 1):
    logger.info(f"\n{i}. {doc['title']} — {doc['section_title']}")
    logger.info(f"   Category: {doc['primary_category']}")
    logger.info(f"   Text preview: {doc['text'][:150]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Building the RAG Prompt

# COMMAND ----------


def build_rag_prompt(question: str, documents: list[dict]) -> str:
    """Build a prompt with retrieved HF doc context.

    Args:
        question: The user's question
        documents: List of retrieved doc chunks

    Returns:
        Formatted prompt string
    """
    context_parts = []
    for i, doc in enumerate(documents, 1):
        context_parts.append(
            f"Document {i}: {doc['title']} — {doc['section_title']}\n"
            f"Content: {doc['text']}"
        )

    context = "\n---\n".join(context_parts)

    return f"""You are a helpful HuggingFace Transformers assistant. \
Answer the question based on the provided documentation context.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer based on the provided documentation
- If the context doesn't contain enough information, say so
- Cite the relevant doc titles and sections when making claims
- Be concise but thorough

ANSWER:"""


# COMMAND ----------

test_prompt = build_rag_prompt("How do I load a pretrained model?", docs)
logger.info("Built RAG prompt:")
logger.info(f"Prompt length: {len(test_prompt)} characters")
logger.info(f"Preview:\n{test_prompt[:500]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. RAG Query Function

# COMMAND ----------


def rag_query(question: str, num_docs: int = 5) -> dict:
    """Answer a question about HuggingFace Transformers using RAG.

    Args:
        question: The user's question
        num_docs: Number of doc chunks to retrieve

    Returns:
        Dictionary with answer and sources
    """
    logger.info(f"Retrieving docs for: '{question}'")
    documents = retrieve_documents(question, num_results=num_docs)
    logger.info(f"Retrieved {len(documents)} chunks")

    prompt = build_rag_prompt(question, documents)

    logger.info("Generating answer...")
    response = client.chat.completions.create(
        model=cfg.llm_endpoint,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.7,
    )

    answer = response.choices[0].message.content

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "title": doc["title"],
                "section": doc["section_title"],
                "category": doc["primary_category"],
            }
            for doc in documents
        ],
    }


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Test RAG System

# COMMAND ----------

result = rag_query("How do I fine-tune a pretrained model with the Trainer API?")

logger.info("=" * 80)
logger.info(f"Question: {result['question']}")
logger.info("=" * 80)
logger.info(f"\nAnswer:\n{result['answer']}")
logger.info("\nSources:")
for src in result["sources"]:
    logger.info(f"  - {src['title']} / {src['section']} ({src['category']})")

# COMMAND ----------

result2 = rag_query("What tokenizers are available in HuggingFace Transformers?")

logger.info("=" * 80)
logger.info(f"Question: {result2['question']}")
logger.info("=" * 80)
logger.info(f"\nAnswer:\n{result2['answer']}")
logger.info("\nSources:")
for src in result2["sources"]:
    logger.info(f"  - {src['title']} / {src['section']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. RAG with Conversation History
# MAGIC
# MAGIC Extend RAG to support multi-turn conversations about HuggingFace Transformers.

# COMMAND ----------


class SimpleRAG:
    """Simple RAG system with conversation history over HF Transformers docs."""

    def __init__(self, llm_endpoint: str, index_name: str) -> None:
        self.llm_endpoint = llm_endpoint
        self.index_name = index_name
        self.conversation_history: list[dict] = []

        self.w = WorkspaceClient()
        _token = (
            dbutils.notebook.entry_point.getDbutils()
            .notebook()
            .getContext()
            .apiToken()
            .get()
        )  # noqa: F821
        self.client = OpenAI(
            api_key=_token,
            base_url=f"{self.w.config.host}/serving-endpoints",
        )
        self.vsc = VectorSearchClient(
            workspace_url=self.w.config.host,
            personal_access_token=_token,
        )

    def retrieve(self, query: str, num_results: int = 5) -> list[dict]:
        """Retrieve relevant HF doc chunks."""
        index = self.vsc.get_index(index_name=self.index_name)
        results = index.similarity_search(
            query_text=query,
            columns=["text", "title", "section_title"],
            num_results=num_results,
            query_type="hybrid",
        )

        documents = []
        if results and "result" in results:
            for row in results["result"].get("data_array", []):
                documents.append(
                    {"text": row[0], "title": row[1], "section_title": row[2]}
                )
        return documents

    def chat(self, question: str, num_docs: int = 3) -> str:
        """Chat with RAG, maintaining conversation history."""
        documents = self.retrieve(question, num_results=num_docs)

        context = "\n\n".join(
            f"[{doc['title']} — {doc['section_title']}]: {doc['text']}"
            for doc in documents
        )

        system_message = (
            "You are a helpful HuggingFace Transformers assistant. "
            "Use the following documentation context to answer questions.\n\n"
            f"CONTEXT:\n{context}\n\n"
            "If the context doesn't contain relevant information, say so. "
            "Always cite doc titles and sections when making claims."
        )

        self.conversation_history.append({"role": "user", "content": question})
        messages = [
            {"role": "system", "content": system_message}
        ] + self.conversation_history

        response = self.client.chat.completions.create(
            model=self.llm_endpoint,
            messages=messages,
            max_tokens=1000,
        )

        answer = response.choices[0].message.content
        self.conversation_history.append({"role": "assistant", "content": answer})
        return answer

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []


# COMMAND ----------

index_name = f"{cfg.catalog}.{cfg.schema}.hf_docs_index"
rag = SimpleRAG(llm_endpoint=cfg.llm_endpoint, index_name=index_name)

logger.info("SimpleRAG initialized")

# COMMAND ----------

logger.info("Starting multi-turn RAG conversation...")
logger.info("=" * 80)

q1 = "What is the pipeline API in HuggingFace Transformers?"
a1 = rag.chat(q1)
logger.info(f"Q: {q1}")
logger.info(f"A: {a1}\n")

# COMMAND ----------

q2 = "What tasks does it support?"
a2 = rag.chat(q2)
logger.info(f"Q: {q2}")
logger.info(f"A: {a2}\n")

# COMMAND ----------

q3 = "How do I use it for text classification?"
a3 = rag.chat(q3)
logger.info(f"Q: {q3}")
logger.info(f"A: {a3}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC In this notebook, we learned:
# MAGIC
# MAGIC 1. What RAG is and why it's useful
# MAGIC 2. How to retrieve HuggingFace doc chunks using Vector Search
# MAGIC 3. How to build prompts with retrieved context
# MAGIC 4. How to combine retrieval and generation
# MAGIC 5. How to build a RAG system with conversation history
# MAGIC
# MAGIC **Key Takeaways:**
# MAGIC - RAG grounds LLM responses in actual documentation
# MAGIC - Vector search enables semantic retrieval over HF Transformers docs
# MAGIC - Context window management is important
# MAGIC - Conversation history enables follow-up questions
# MAGIC
# MAGIC **Next**: Lecture 3.2 - MCP Integration
