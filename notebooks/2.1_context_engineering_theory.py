# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 2.1: Context Engineering Theory
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - What is context engineering
# MAGIC - RAG (Retrieval Augmented Generation) fundamentals
# MAGIC - Context window limitations
# MAGIC - Strategies for effective context engineering
# MAGIC - Trade-offs and best practices

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. What is Context Engineering?
# MAGIC
# MAGIC **Context Engineering** is the practice of providing relevant information to LLMs to improve their responses.
# MAGIC
# MAGIC ### Why Context Matters
# MAGIC
# MAGIC - LLMs have knowledge cutoff dates
# MAGIC - Need access to private/proprietary data
# MAGIC - Reduce hallucinations
# MAGIC - Improve accuracy and relevance
# MAGIC - Enable domain-specific applications
# MAGIC
# MAGIC ### Our Use Case: HuggingFace Transformers Documentation
# MAGIC
# MAGIC We're building a RAG agent over the official HuggingFace Transformers docs.
# MAGIC This lets users ask natural-language questions like:
# MAGIC - "How do I fine-tune a model with LoRA?"
# MAGIC - "What's the difference between AutoModel and AutoModelForCausalLM?"
# MAGIC - "How do I use a custom tokenizer?"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. RAG (Retrieval Augmented Generation)
# MAGIC
# MAGIC ### RAG Pipeline Overview
# MAGIC
# MAGIC ```
# MAGIC User Query
# MAGIC     |
# MAGIC Query Embedding
# MAGIC     |
# MAGIC Vector Search (Retrieve relevant doc chunks)
# MAGIC     |
# MAGIC Context Assembly
# MAGIC     |
# MAGIC LLM Prompt (Query + Context)
# MAGIC     |
# MAGIC Generated Response
# MAGIC ```
# MAGIC
# MAGIC ### Benefits of RAG
# MAGIC
# MAGIC 1. **Up-to-date Information**: HF Transformers docs change with every release
# MAGIC 2. **Domain Knowledge**: Deep library-specific information
# MAGIC 3. **Reduced Hallucinations**: Grounded in retrieved documentation
# MAGIC 4. **Cost-Effective**: No need to fine-tune for every docs update
# MAGIC 5. **Transparency**: Can cite source doc pages

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Context Window Limitations
# MAGIC
# MAGIC ### Token Limits by Model
# MAGIC
# MAGIC | Model | Context Window | Notes |
# MAGIC |-------|---------------|-------|
# MAGIC | GPT-3.5 Turbo | 16K tokens | ~12K words |
# MAGIC | GPT-4 | 8K-128K tokens | Varies by version |
# MAGIC | Claude 3 | 200K tokens | ~150K words |
# MAGIC | Llama 3.1 70B | 128K tokens | ~96K words |
# MAGIC | Llama 4 Maverick | 1M tokens | Our primary model |
# MAGIC | Gemini 1.5 Pro | 1M tokens | ~750K words |

# COMMAND ----------

from loguru import logger

# COMMAND ----------


# Example: Token estimation
def estimate_tokens(text: str) -> int:
    """Rough estimation: ~4 characters per token."""
    return len(text) // 4


# Example texts
short_text = "How do I fine-tune a BERT model?"
long_text = (
    "The HuggingFace Transformers library provides thousands of "
    "pretrained models for NLP, vision, and audio tasks. "
) * 100

logger.info(f"Short text: {estimate_tokens(short_text)} tokens")
logger.info(f"Long text: {estimate_tokens(long_text)} tokens")

# Context window example
context_window = 1_000_000  # Llama 4 Maverick
system_prompt_tokens = 200
user_query_tokens = 100
max_output_tokens = 2000

available = context_window - system_prompt_tokens - user_query_tokens - max_output_tokens
logger.info(f"\nAvailable tokens for context: {available:,}")
logger.info(f"Approximate words: {available * 0.75:,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Retrieval Strategies Overview
# MAGIC
# MAGIC ### Three Main Approaches
# MAGIC
# MAGIC **1. Semantic Search (Vector Search)**
# MAGIC - Convert documents to embeddings (numerical vectors)
# MAGIC - Find documents with similar vector representations
# MAGIC - **Pros**: Captures meaning, handles synonyms, cross-lingual
# MAGIC - **Cons**: May miss exact keyword matches
# MAGIC
# MAGIC **2. Hybrid Search**
# MAGIC - Combines semantic search (embeddings) + keyword search (BM25)
# MAGIC - Merges results using fusion algorithms
# MAGIC - **Pros**: Best of both worlds — meaning + exact matches
# MAGIC - **Cons**: More complex, slightly slower
# MAGIC
# MAGIC **3. Reranking**
# MAGIC - Retrieve more candidates (e.g., top 20-50)
# MAGIC - Use a cross-encoder model to rerank
# MAGIC - Return top-k after reranking
# MAGIC - **Pros**: Higher precision
# MAGIC - **Cons**: Additional computation cost
# MAGIC
# MAGIC **Note**: We'll implement these with actual code in **Notebook 2.4**.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query Enhancement Technique: Query Rewriting
# MAGIC
# MAGIC Before retrieval, enhance the query by generating variations:
# MAGIC - Use different terminology
# MAGIC - Make it more specific or general
# MAGIC - Focus on different aspects
# MAGIC
# MAGIC This improves recall by searching with multiple phrasings.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from openai import OpenAI

w = WorkspaceClient()

# Authenticate using Databricks SDK
host = w.config.host
token = w.tokens.create(lifetime_seconds=1200).token_value

client = OpenAI(
    api_key=token,
    base_url=f"{host.rstrip('/')}/serving-endpoints",
)

MODEL_NAME = "databricks-llama-4-maverick"
logger.info(f"Using model: {MODEL_NAME}")


def rewrite_query(original_query: str) -> list[str]:
    """Generate query variations for better retrieval."""

    prompt = f"""Given this search query about HuggingFace Transformers, generate 3 alternative phrasings that would help retrieve relevant documentation:

Original query: {original_query}

Generate 3 variations that:
1. Use different terminology
2. Are more specific or more general
3. Focus on different aspects of the library

Return only the 3 variations, one per line."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.7,
    )

    variations = response.choices[0].message.content.strip().split("\n")
    return [v.strip() for v in variations if v.strip()]


# Example
original = "How do I load a pretrained model?"
variations = rewrite_query(original)

logger.info(f"Original: {original}\n")
logger.info("Variations:")
for i, var in enumerate(variations, 1):
    logger.info(f"{i}. {var}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Context Quality vs Quantity
# MAGIC
# MAGIC ### The Trade-off
# MAGIC
# MAGIC - **More context** = More information, but:
# MAGIC   - Higher cost (more tokens)
# MAGIC   - Slower inference
# MAGIC   - Risk of "lost in the middle" problem
# MAGIC   - More noise
# MAGIC
# MAGIC - **Less context** = Faster and cheaper, but:
# MAGIC   - May miss important information
# MAGIC   - Less comprehensive answers

# COMMAND ----------


# Context ordering to avoid "lost in the middle" problem
def order_context_by_relevance(chunks: list[dict]) -> list[dict]:
    """Order chunks: most relevant at start, second-most at end, rest in middle."""
    if len(chunks) <= 2:
        return chunks

    ordered = [chunks[0]]  # Most relevant at start
    if len(chunks) > 2:
        ordered.extend(chunks[2:-1])  # Less relevant in middle
    if len(chunks) > 1:
        ordered.append(chunks[1])  # Second most relevant at end
    return ordered


chunks = [
    {"text": "AutoModel.from_pretrained() loads any model", "score": 0.95},
    {"text": "Tokenizer handles text preprocessing", "score": 0.88},
    {"text": "Pipeline API simplifies inference", "score": 0.75},
    {"text": "Config objects control architecture", "score": 0.70},
    {"text": "Trainer handles the training loop", "score": 0.65},
]

ordered = order_context_by_relevance(chunks)
logger.info("Ordered chunks (most relevant at start/end):")
for i, chunk in enumerate(ordered, 1):
    logger.info(f"{i}. {chunk['text']} (score: {chunk['score']})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Context Compression Techniques
# MAGIC
# MAGIC When context is too large for the window, compress it:
# MAGIC
# MAGIC ### Technique 1: Extractive Summarization
# MAGIC - Select most important sentences/chunks
# MAGIC - Fast, preserves original wording
# MAGIC
# MAGIC ### Technique 2: Abstractive Summarization
# MAGIC - Use LLM to generate summaries
# MAGIC - More concise and coherent
# MAGIC - Slower, costs tokens

# COMMAND ----------


def summarize_chunk(text: str, max_length: int = 100) -> str:
    """Summarize a text chunk using LLM."""

    prompt = f"""Summarize the following HuggingFace Transformers documentation in {max_length} words or less, preserving key technical details:

{text}

Summary:"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_length * 2,
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()


long_text = """
The Trainer class provides an API for feature-complete training in PyTorch,
and it supports distributed training on multiple GPUs/TPUs, mixed precision
through NVIDIA Apex and native PyTorch AMP. The Trainer contains the basic
training loop which supports gradient accumulation, gradient clipping,
logging, evaluation and prediction. It also integrates with Weights & Biases,
TensorBoard, and MLflow for experiment tracking.
"""

summary = summarize_chunk(long_text, max_length=50)
logger.info(f"Original length: {len(long_text)} chars")
logger.info(f"Summary length: {len(summary)} chars")
logger.info(f"\nSummary:\n{summary}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Metadata Filtering
# MAGIC
# MAGIC Enhance retrieval with metadata filters:
# MAGIC
# MAGIC - **Category filters**: "Get started" vs "Task guides" vs "Optimization"
# MAGIC - **Section filters**: Specific doc sections
# MAGIC - **Recency**: Filter by ingestion date

# COMMAND ----------

import json

# Example: Metadata structure for an HF doc chunk
example_document = {
    "id": "quicktour_0",
    "text": "Pipeline groups together a pretrained model with...",
    "doc_id": "quicktour",
    "section_title": "Pipeline",
    "title": "Quick tour",
    "primary_category": "Get started",
    "url": "https://huggingface.co/docs/transformers/quicktour",
}

logger.info("Example document with metadata:")
logger.info(json.dumps(example_document, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Prompt Engineering for RAG
# MAGIC
# MAGIC ### Effective RAG Prompts
# MAGIC
# MAGIC ```
# MAGIC System: You are a helpful AI assistant that answers questions about
# MAGIC HuggingFace Transformers based on the official documentation.
# MAGIC If the answer is not in the context, say so.
# MAGIC
# MAGIC Context:
# MAGIC [Retrieved doc chunks here]
# MAGIC
# MAGIC Question: [User question]
# MAGIC ```

# COMMAND ----------


def create_rag_prompt(query: str, context_chunks: list[str]) -> str:
    """Create a RAG prompt with context."""

    context = "\n\n".join(
        [f"[Document {i + 1}]\n{chunk}" for i, chunk in enumerate(context_chunks)]
    )

    return f"""Use the following HuggingFace Transformers documentation to answer the question. If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}

Answer:"""


query = "How do I load a pretrained model?"
context_chunks = [
    "AutoModel.from_pretrained('bert-base-uncased') loads a pretrained BERT model.",
    "The from_pretrained() method downloads the model weights from the Hub.",
]

prompt = create_rag_prompt(query, context_chunks)
logger.info(prompt)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Evaluation Metrics for Context Engineering
# MAGIC
# MAGIC ### Retrieval Metrics
# MAGIC - **Precision@K**: Proportion of retrieved docs that are relevant
# MAGIC - **Recall@K**: Proportion of relevant docs that are retrieved
# MAGIC - **MRR (Mean Reciprocal Rank)**: Position of first relevant result
# MAGIC - **NDCG**: Ranking quality
# MAGIC
# MAGIC ### End-to-End Metrics
# MAGIC - **Answer Relevance**: Is the answer relevant to the question?
# MAGIC - **Faithfulness**: Is the answer grounded in the context?
# MAGIC - **Context Relevance**: Is the retrieved context relevant?

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Best Practices Summary
# MAGIC
# MAGIC ### Do:
# MAGIC 1. Chunk documents appropriately (more in next notebook)
# MAGIC 2. Use high-quality embeddings (databricks-gte-large-en)
# MAGIC 3. Implement metadata filtering by doc category
# MAGIC 4. Order context strategically
# MAGIC 5. Monitor retrieval quality
# MAGIC 6. Provide clear instructions in prompts
# MAGIC 7. Handle cases where context doesn't contain the answer
# MAGIC
# MAGIC ### Don't:
# MAGIC 1. Exceed context window limits
# MAGIC 2. Include irrelevant information
# MAGIC 3. Ignore the "lost in the middle" problem
# MAGIC 4. Forget to cite source doc pages
# MAGIC 5. Assume all retrieved docs are relevant
