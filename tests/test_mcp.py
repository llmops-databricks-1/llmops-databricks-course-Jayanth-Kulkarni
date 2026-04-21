"""Tests for stackoverflow_curator.mcp module."""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from stackoverflow_curator.mcp import ToolInfo, create_mcp_tools


class TestToolInfo:
    def test_fields(self) -> None:
        def noop(**kwargs: str) -> str:
            return "ok"

        tool = ToolInfo(
            name="my_tool",
            spec={"type": "function"},
            exec_fn=noop,
        )
        assert tool.name == "my_tool"
        assert tool.spec == {"type": "function"}
        assert tool.exec_fn() == "ok"


class TestCreateMcpTools:
    def test_empty_urls_returns_empty(self) -> None:
        w = MagicMock()
        with patch("databricks_mcp.DatabricksMCPClient"):
            result = create_mcp_tools(w, [])
        assert result == []

    def test_failed_url_returns_empty(self) -> None:
        w = MagicMock()
        with patch(
            "databricks_mcp.DatabricksMCPClient",
            side_effect=RuntimeError("connection failed"),
        ):
            result = create_mcp_tools(w, ["http://bad-url"])
        assert result == []

    def test_tools_loaded_from_mcp_server(self) -> None:
        @dataclass
        class FakeTool:
            name: str = "search_tool"
            description: str = "Searches docs"
            inputSchema: dict = None  # noqa: N815

            def __post_init__(self) -> None:
                if self.inputSchema is None:
                    self.inputSchema = {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    }

        mock_client = MagicMock()
        mock_client.list_tools.return_value = [FakeTool()]
        fake_content = MagicMock()
        fake_content.text = "result text"
        mock_client.call_tool.return_value = MagicMock(content=[fake_content])

        w = MagicMock()
        with patch("databricks_mcp.DatabricksMCPClient", return_value=mock_client):
            tools = create_mcp_tools(w, ["http://fake-mcp-url"])

        assert len(tools) == 1
        assert tools[0].name == "search_tool"
        assert tools[0].spec["type"] == "function"
        assert tools[0].spec["function"]["name"] == "search_tool"

    def test_exec_fn_calls_tool(self) -> None:
        @dataclass
        class FakeTool:
            name: str = "my_tool"
            description: str = "A tool"
            inputSchema: dict = None  # noqa: N815

            def __post_init__(self) -> None:
                if self.inputSchema is None:
                    self.inputSchema = {"type": "object", "properties": {}}

        mock_client = MagicMock()
        mock_client.list_tools.return_value = [FakeTool()]
        fake_content = MagicMock()
        fake_content.text = "tool output"
        mock_client.call_tool.return_value = MagicMock(content=[fake_content])

        w = MagicMock()
        with patch("databricks_mcp.DatabricksMCPClient", return_value=mock_client):
            tools = create_mcp_tools(w, ["http://fake"])

        result = tools[0].exec_fn(query="test query")
        mock_client.call_tool.assert_called_once_with("my_tool", {"query": "test query"})
        assert result == "tool output"
