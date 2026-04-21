# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 3.1: Custom Functions & Tools for Agents
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - What are agent tools?
# MAGIC - Creating custom functions
# MAGIC - Tool specifications (OpenAI format)
# MAGIC - Integrating tools with agents
# MAGIC - Vector search as a tool

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Understanding Agent Tools
# MAGIC
# MAGIC **Tools** are functions that agents can call to perform specific tasks.
# MAGIC
# MAGIC ### Why Tools?
# MAGIC
# MAGIC LLMs alone cannot:
# MAGIC - Access external data (databases, APIs)
# MAGIC - Perform calculations
# MAGIC - Execute code
# MAGIC - Search documents
# MAGIC
# MAGIC **Tools bridge this gap** by giving LLMs the ability to take actions.
# MAGIC
# MAGIC ### Tool Calling Flow:
# MAGIC
# MAGIC ```
# MAGIC User: "How do I fine-tune a model?"
# MAGIC   ↓
# MAGIC Agent: Decides to use search_hf_docs tool
# MAGIC   ↓
# MAGIC Tool: search_hf_docs(query="fine-tune model")
# MAGIC   ↓
# MAGIC Tool Result: [doc_chunk1, doc_chunk2, doc_chunk3]
# MAGIC   ↓
# MAGIC Agent: Synthesizes answer from results
# MAGIC   ↓
# MAGIC Response: "To fine-tune a model, you can..."
# MAGIC ```

# COMMAND ----------

import json
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
from loguru import logger
from openai import OpenAI
from pyspark.sql import SparkSession

from stackoverflow_curator.config import get_env, load_config
from stackoverflow_curator.mcp import ToolInfo

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

env = get_env(spark)
cfg = load_config("../project_config.yml", env)

w = WorkspaceClient()
_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()  # noqa: F821
vsc = VectorSearchClient(
    workspace_url=w.config.host,
    personal_access_token=_token,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Tool Specification Format
# MAGIC
# MAGIC Tools are defined using the **OpenAI function calling format**:
# MAGIC
# MAGIC ```json
# MAGIC {
# MAGIC   "type": "function",
# MAGIC   "function": {
# MAGIC     "name": "tool_name",
# MAGIC     "description": "What the tool does",
# MAGIC     "parameters": {
# MAGIC       "type": "object",
# MAGIC       "properties": {
# MAGIC         "param1": {
# MAGIC           "type": "string",
# MAGIC           "description": "Description of param1"
# MAGIC         }
# MAGIC       },
# MAGIC       "required": ["param1"]
# MAGIC     }
# MAGIC   }
# MAGIC }
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Creating a Simple Calculator Tool

# COMMAND ----------


def calculator(operation: str, a: float, b: float) -> float:
    """Perform basic arithmetic operations.

    Args:
        operation: One of 'add', 'subtract', 'multiply', 'divide'
        a: First number
        b: Second number

    Returns:
        Result of the operation
    """
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else float("inf"),
    }

    if operation not in operations:
        raise ValueError(f"Unknown operation: {operation}")

    return operations[operation](a, b)


result = calculator("multiply", 5, 3)
logger.info(f"5 * 3 = {result}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tool Specification for Calculator

# COMMAND ----------

calculator_tool_spec = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Perform basic arithmetic operations (add, subtract, multiply, divide)",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The arithmetic operation to perform",
                },
                "a": {"type": "number", "description": "The first number"},
                "b": {"type": "number", "description": "The second number"},
            },
            "required": ["operation", "a", "b"],
        },
    },
}

logger.info("Calculator Tool Specification:")
logger.info(json.dumps(calculator_tool_spec, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Creating a Vector Search Tool over HuggingFace Docs

# COMMAND ----------


def parse_vector_search_results(results: dict) -> list[dict]:
    """Parse vector search results from array format to dict format."""
    columns = [col["name"] for col in results.get("manifest", {}).get("columns", [])]
    data_array = results.get("result", {}).get("data_array", [])
    return [dict(zip(columns, row_data, strict=False)) for row_data in data_array]


# COMMAND ----------


def search_hf_docs(
    query: str,
    num_results: int = 5,
    category_filter: str | None = None,
) -> str:
    """Search HuggingFace Transformers documentation using vector search.

    Args:
        query: Search query
        num_results: Number of results to return
        category_filter: Optional category filter (e.g., 'Get started', 'Tutorials')

    Returns:
        JSON string with search results
    """
    index_name = f"{cfg.catalog}.{cfg.schema}.hf_docs_index"
    index = vsc.get_index(index_name=index_name)

    search_params: dict[str, Any] = {
        "query_text": query,
        "columns": ["text", "id", "title", "section_title", "primary_category"],
        "num_results": num_results,
        "query_type": "hybrid",
    }

    if category_filter:
        search_params["filters"] = {"primary_category": category_filter}

    results = index.similarity_search(**search_params)

    chunks = []
    for row in parse_vector_search_results(results):
        chunks.append(
            {
                "title": row.get("title", "N/A"),
                "section": row.get("section_title", "N/A"),
                "category": row.get("primary_category", "N/A"),
                "excerpt": row.get("text", "")[:300] + "...",
            }
        )

    return json.dumps(chunks, indent=2)


test_result = search_hf_docs("how to load a pretrained model", num_results=2)
logger.info("Search Results:")
logger.info(test_result)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tool Specification for HF Docs Search

# COMMAND ----------

search_hf_docs_tool_spec = {
    "type": "function",
    "function": {
        "name": "search_hf_docs",
        "description": (
            "Search the HuggingFace Transformers documentation using semantic search. "
            "Returns relevant documentation chunks with titles and excerpts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query describing what documentation to find",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5)",
                    "default": 5,
                },
                "category_filter": {
                    "type": "string",
                    "description": (
                        "Optional category filter, e.g. 'Get started', 'Tutorials', "
                        "'API', 'How-To'"
                    ),
                },
            },
            "required": ["query"],
        },
    },
}

logger.info("HF Docs Search Tool Specification:")
logger.info(json.dumps(search_hf_docs_tool_spec, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Tool Information Class

# COMMAND ----------

calculator_tool = ToolInfo(
    name="calculator",
    spec=calculator_tool_spec,
    exec_fn=calculator,
)

search_hf_docs_tool = ToolInfo(
    name="search_hf_docs",
    spec=search_hf_docs_tool_spec,
    exec_fn=search_hf_docs,
)

logger.info("Available Tools:")
logger.info(f"1. {calculator_tool.name}")
logger.info(f"2. {search_hf_docs_tool.name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Tool Registry Pattern

# COMMAND ----------


class ToolRegistry:
    """Registry for managing agent tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolInfo] = {}

    def register(self, tool: ToolInfo) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get_tool(self, name: str) -> ToolInfo:
        """Get a tool by name."""
        if name not in self._tools:
            raise ValueError(f"Tool not found: {name}")
        return self._tools[name]

    def get_all_specs(self) -> list[dict]:
        """Get all tool specifications."""
        return [tool.spec for tool in self._tools.values()]

    def execute(self, name: str, args: dict) -> Any:
        """Execute a tool with arguments."""
        tool = self.get_tool(name)
        return tool.exec_fn(**args)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())


registry = ToolRegistry()
registry.register(calculator_tool)
registry.register(search_hf_docs_tool)

logger.info(f"Total tools registered: {len(registry.list_tools())}")
logger.info(f"Tools: {registry.list_tools()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Executing Tools

# COMMAND ----------

calc_result = registry.execute("calculator", {"operation": "add", "a": 10, "b": 5})
logger.info(f"Calculator result: {calc_result}")

search_result = registry.execute(
    "search_hf_docs", {"query": "tokenizer encode decode", "num_results": 3}
)
logger.info(f"Search result:\n{search_result}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Best Practices for Tool Design
# MAGIC
# MAGIC ### Do:
# MAGIC 1. **Clear descriptions**: Help the LLM understand when to use the tool
# MAGIC 2. **Type hints**: Use proper Python type hints
# MAGIC 3. **Error handling**: Handle errors gracefully
# MAGIC 4. **Return structured data**: JSON or clear text format
# MAGIC 5. **Validate inputs**: Check parameters before execution
# MAGIC 6. **Document parameters**: Clear parameter descriptions
# MAGIC
# MAGIC ### Don't:
# MAGIC 1. Create tools that are too complex
# MAGIC 2. Return unstructured or ambiguous data
# MAGIC 3. Forget error handling
# MAGIC 4. Make tools that take too long to execute
# MAGIC 5. Overlap tool functionality
# MAGIC 6. Use unclear tool names

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Testing Tools

# COMMAND ----------


def test_tool(tool_name: str, test_cases: list[dict]) -> None:
    """Test a tool with multiple test cases."""
    logger.info(f"Testing tool: {tool_name}")
    logger.info("=" * 80)

    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"Test Case {i}:")
        logger.info(f"  Input: {test_case}")

        try:
            result = registry.execute(tool_name, test_case)
            logger.info("  Success")
            logger.info(f"  Result: {str(result)[:100]}...")
        except Exception as e:
            logger.error(f"  Error: {e}")


test_tool(
    "calculator",
    [
        {"operation": "add", "a": 5, "b": 3},
        {"operation": "multiply", "a": 4, "b": 7},
        {"operation": "divide", "a": 10, "b": 2},
    ],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Using Tools with an Agent

# COMMAND ----------


class SimpleAgent:
    """A simple agent that can call tools in a loop."""

    def __init__(
        self, llm_endpoint: str, system_prompt: str, tools: list[ToolInfo]
    ) -> None:
        self.llm_endpoint = llm_endpoint
        self.system_prompt = system_prompt
        self._tools_dict = {tool.name: tool for tool in tools}
        self._client = OpenAI(
            api_key=dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get(),  # noqa: F821
            base_url=f"{w.config.host}/serving-endpoints",
        )

    def get_tool_specs(self) -> list[dict]:
        """Get tool specifications for the LLM."""
        return [tool.spec for tool in self._tools_dict.values()]

    def execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute a tool by name."""
        if tool_name not in self._tools_dict:
            raise ValueError(f"Unknown tool: {tool_name}")
        return self._tools_dict[tool_name].exec_fn(**args)

    def chat(self, user_message: str, max_iterations: int = 10) -> str:
        """Chat with the agent, allowing tool calls."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        for _iteration in range(max_iterations):
            response = self._client.chat.completions.create(
                model=self.llm_endpoint,
                messages=messages,
                tools=self.get_tool_specs() if self._tools_dict else None,
            )

            assistant_message = response.choices[0].message

            if assistant_message.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in assistant_message.tool_calls
                        ],
                    }
                )

                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    logger.info(f"Calling tool: {tool_name}({tool_args})")
                    try:
                        result = self.execute_tool(tool_name, tool_args)
                    except Exception as e:
                        result = f"Error: {e!s}"

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(result),
                        }
                    )
            else:
                return assistant_message.content

        return "Max iterations reached."


# COMMAND ----------

agent = SimpleAgent(
    llm_endpoint=cfg.llm_endpoint,
    system_prompt=(
        "You are a helpful HuggingFace Transformers assistant. "
        "Use the available tools to search documentation and answer questions."
    ),
    tools=[calculator_tool, search_hf_docs_tool],
)

logger.info("Agent created with tools:")
for tool_name in agent._tools_dict:
    logger.info(f"  - {tool_name}")

# COMMAND ----------

logger.info("Testing agent with calculator:")
logger.info("=" * 80)
response = agent.chat("What is 42 multiplied by 17?")
logger.info(f"Agent response: {response}")

# COMMAND ----------

logger.info("Testing agent with docs search:")
logger.info("=" * 80)
response = agent.chat("How do I load a pretrained BERT model with from_pretrained?")
logger.info(f"Agent response: {response}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC In this notebook, we learned:
# MAGIC
# MAGIC 1. What agent tools are and why they're important
# MAGIC 2. Tool specification format (OpenAI function calling)
# MAGIC 3. Creating custom functions as tools
# MAGIC 4. Vector search over HuggingFace docs as a tool
# MAGIC 5. Tool registry pattern for managing tools
# MAGIC 6. Executing tools programmatically
# MAGIC 7. Best practices for tool design
# MAGIC 8. Testing tools
# MAGIC 9. Building a simple agent with tool calling loop
# MAGIC
# MAGIC **Next**: Lecture 3.1b - Simple RAG
