"""Tool definitions for agent tool calling."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from databricks.sdk import WorkspaceClient


@dataclass
class ToolInfo:
    """Represents an agent tool with its specification and execution function."""

    name: str
    spec: dict
    exec_fn: Callable[..., Any]


def create_mcp_tools(w: WorkspaceClient, mcp_urls: list[str]) -> list[ToolInfo]:
    """Convert MCP server tools into agent-compatible ToolInfo objects.

    Args:
        w: Databricks WorkspaceClient
        mcp_urls: List of MCP server URLs to load tools from

    Returns:
        List of ToolInfo objects ready for use with an agent
    """
    import logging

    from databricks_mcp import DatabricksMCPClient

    tools = []
    for url in mcp_urls:
        mcp_tool_list = []
        try:
            client = DatabricksMCPClient(server_url=url, workspace_client=w)
            mcp_tool_list = client.list_tools()
        except Exception as e:
            logging.error(f"MCP tool load failed for {url}: {type(e).__name__}: {e}")

        for mcp_tool in mcp_tool_list:
            spec = {
                "type": "function",
                "function": {
                    "name": mcp_tool.name,
                    "description": mcp_tool.description or "",
                    "parameters": mcp_tool.inputSchema
                    or {"type": "object", "properties": {}},
                },
            }

            def _make_exec_fn(
                _client: DatabricksMCPClient, _name: str
            ) -> Callable[..., str]:
                def exec_fn(**kwargs: str) -> str:
                    result = _client.call_tool(_name, kwargs)
                    return "\n".join(c.text for c in result.content if hasattr(c, "text"))

                return exec_fn

            tools.append(
                ToolInfo(
                    name=mcp_tool.name,
                    spec=spec,
                    exec_fn=_make_exec_fn(client, mcp_tool.name),
                )
            )

    return tools
