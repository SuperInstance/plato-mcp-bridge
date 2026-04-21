"""MCP bridge — Model Context Protocol integration for PLATO.
Part of the PLATO framework."""
from .mcp import McpBridge, McpTool, McpContext, McpMiddleware, LoggingMiddleware, ToolStatus
__version__ = "0.2.0"
__all__ = ["McpBridge", "McpTool", "McpContext", "McpMiddleware", "LoggingMiddleware", "ToolStatus"]
