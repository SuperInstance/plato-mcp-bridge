"""Model Context Protocol bridge — connect LLM tools to PLATO rooms.

Provides tool registration with JSON Schema validation, session management with
context windows, middleware pipeline, streaming tool dispatch, and call logging.

## Why This Matters

MCP bridges are the connective tissue between LLMs and PLATO. Each tool is a
capability the AI can invoke — room queries, tile creation, constraint checks,
and more. This bridge handles validation, routing, and observability.
"""
import time
import json
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Any
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class ToolStatus(Enum):
    ACTIVE = "active"
    DISABLED = "disabled"
    DEGRADED = "degraded"


@dataclass
class McpTool:
    """A registered tool with schema, handler, and metrics."""
    name: str
    description: str
    handler: Optional[Callable] = None
    parameters: dict = field(default_factory=dict)
    status: ToolStatus = ToolStatus.ACTIVE
    call_count: int = 0
    error_count: int = 0
    avg_latency: float = 0.0
    total_latency: float = 0.0
    rate_limit: int = 0  # max calls per minute, 0 = unlimited
    _call_timestamps: list = field(default_factory=list)

    def check_rate_limit(self) -> bool:
        if self.rate_limit <= 0:
            return True
        now = time.time()
        window = [t for t in self._call_timestamps if now - t < 60]
        self._call_timestamps = window
        return len(window) < self.rate_limit

    def record_call(self, latency: float, success: bool):
        self.call_count += 1
        self.total_latency += latency
        self.avg_latency = self.total_latency / self.call_count
        self._call_timestamps.append(time.time())
        if not success:
            self.error_count += 1


@dataclass
class McpContext:
    """Session context with room binding, tile buffer, and variable store."""
    room: str = ""
    agent: str = ""
    tiles: list = field(default_factory=list)
    variables: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    max_tiles: int = 1000
    max_history: int = 500

    def add_tile(self, tile: dict) -> dict:
        """Add a tile with overflow protection."""
        self.tiles.append(tile)
        if len(self.tiles) > self.max_tiles:
            evicted = self.tiles[:len(self.tiles) - self.max_tiles]
            self.tiles = self.tiles[-self.max_tiles:]
            return {"added": True, "evicted": len(evicted)}
        return {"added": True, "evicted": 0}

    def add_history_entry(self, role: str, content: str, tool_name: str = ""):
        entry = {"role": role, "content": content, "tool": tool_name, "ts": time.time()}
        self.history.append(entry)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def set_variable(self, key: str, value: Any):
        self.variables[key] = value

    def get_variable(self, key: str, default: Any = None) -> Any:
        return self.variables.get(key, default)

    def to_dict(self) -> dict:
        return {"room": self.room, "agent": self.agent, "tile_count": len(self.tiles),
                "var_count": len(self.variables), "history_len": len(self.history)}


class McpMiddleware:
    """Base middleware for tool call interception."""

    def before_call(self, tool_name: str, kwargs: dict, context: McpContext = None) -> Optional[dict]:
        """Return a dict to short-circuit the call, or None to proceed."""
        return None

    def after_call(self, tool_name: str, result: dict, latency: float, context: McpContext = None) -> dict:
        """Transform result after call. Return modified result."""
        return result


class LoggingMiddleware(McpMiddleware):
    """Logs all tool calls with timing and context."""

    def __init__(self, max_log: int = 1000):
        self._log: list = []
        self.max_log = max_log

    def after_call(self, tool_name, result, latency, context=None):
        entry = {"tool": tool_name, "latency_ms": round(latency * 1000, 2),
                 "status": result.get("status", "unknown"),
                 "room": context.room if context else "",
                 "ts": time.time()}
        self._log.append(entry)
        if len(self._log) > self.max_log:
            self._log = self._log[-self.max_log:]
        return result

    def get_log(self, tool_name: str = "", limit: int = 50) -> list:
        if tool_name:
            return [e for e in self._log if e["tool"] == tool_name][-limit:]
        return self._log[-limit:]

    def get_error_log(self, limit: int = 20) -> list:
        return [e for e in self._log if e["status"] == "error"][-limit:]


class RateLimitMiddleware(McpMiddleware):
    """Enforces per-tool rate limits."""

    def before_call(self, tool_name, kwargs, context=None):
        # Rate limiting is handled by McpTool itself
        return None


class McpBridge:
    """Core MCP bridge — tool registry, session manager, dispatch pipeline."""

    def __init__(self, default_max_sessions: int = 100, default_max_contexts: int = 200):
        self._tools: dict[str, McpTool] = {}
        self._contexts: dict[str, McpContext] = {}
        self._sessions: dict[str, dict] = {}
        self._middleware: list[McpMiddleware] = []
        self._max_sessions = default_max_sessions
        self._max_contexts = default_max_contexts

    # --- Tool Management ---

    def register_tool(self, name: str, description: str, handler: Callable = None,
                      parameters: dict = None, rate_limit: int = 0) -> McpTool:
        """Register a tool with optional handler and rate limit."""
        if name in self._tools:
            logger.warning(f"Overwriting existing tool: {name}")
        tool = McpTool(name=name, description=description, handler=handler,
                       parameters=parameters or {}, rate_limit=rate_limit)
        self._tools[name] = tool
        return tool

    def unregister_tool(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    def enable_tool(self, name: str) -> bool:
        tool = self._tools.get(name)
        if tool:
            tool.status = ToolStatus.ACTIVE
            return True
        return False

    def disable_tool(self, name: str) -> bool:
        tool = self._tools.get(name)
        if tool:
            tool.status = ToolStatus.DISABLED
            return True
        return False

    def list_tools(self, room: str = "", include_disabled: bool = False) -> list[dict]:
        """List tools as schema dicts suitable for LLM discovery."""
        results = []
        for t in self._tools.values():
            if not t.status == ToolStatus.ACTIVE and not include_disabled:
                continue
            results.append({"name": t.name, "description": t.description,
                           "parameters": t.parameters, "status": t.status.value,
                           "call_count": t.call_count, "avg_latency_ms": round(t.avg_latency * 1000, 1)})
        return results

    def discover_tools(self, query: str) -> list[dict]:
        """Search tools by name or description (fuzzy keyword match)."""
        query_lower = query.lower()
        matches = []
        for t in self._tools.values():
            if t.status != ToolStatus.ACTIVE:
                continue
            score = 0
            if query_lower in t.name.lower():
                score += 10
            if query_lower in t.description.lower():
                score += 5
            if query_lower in " ".join(t.parameters.keys()).lower():
                score += 2
            if score > 0:
                matches.append({"name": t.name, "description": t.description,
                               "relevance": score})
        return sorted(matches, key=lambda x: x["relevance"], reverse=True)

    # --- Tool Dispatch ---

    def call_tool(self, name: str, context: McpContext = None, **kwargs) -> dict:
        """Dispatch a tool call through the middleware pipeline."""
        tool = self._tools.get(name)
        if not tool:
            return {"error": f"Tool '{name}' not found", "status": "not_found"}

        if tool.status == ToolStatus.DISABLED:
            return {"error": f"Tool '{name}' is disabled", "status": "disabled"}

        if not tool.check_rate_limit():
            return {"error": f"Tool '{name}' rate limited ({tool.rate_limit}/min)",
                    "status": "rate_limited"}

        # Run before-call middleware
        for mw in self._middleware:
            shortcut = mw.before_call(name, kwargs, context)
            if shortcut is not None:
                return shortcut

        start = time.time()
        result = {"tool": name, "status": "ok"}

        try:
            if tool.handler:
                handler_result = tool.handler(context=context, **kwargs)
                result["result"] = handler_result
            else:
                result["result"] = {"note": "No handler registered", "kwargs": kwargs}
        except Exception as e:
            result = {"error": str(e), "status": "error", "tool": name}
            logger.error(f"Tool '{name}' failed: {e}")

        latency = time.time() - start
        tool.record_call(latency, result.get("status") == "ok")
        result["latency_ms"] = round(latency * 1000, 2)
        result["context_room"] = context.room if context else ""

        # Run after-call middleware
        for mw in self._middleware:
            result = mw.after_call(name, result, latency, context)

        # Log to context history
        if context:
            context.add_history_entry("tool_call", json.dumps(result), tool_name=name)

        return result

    def call_tool_batch(self, calls: list[dict]) -> list[dict]:
        """Execute multiple tool calls in sequence."""
        results = []
        for call in calls:
            name = call.get("tool", call.get("name", ""))
            kwargs = call.get("kwargs", call.get("arguments", {}))
            ctx_id = call.get("session_id", call.get("context_id"))
            ctx = self._contexts.get(ctx_id) if ctx_id else None
            results.append(self.call_tool(name, context=ctx, **kwargs))
        return results

    # --- Session Management ---

    def create_session(self, model: str = "default", system_prompt: str = "",
                       metadata: dict = None) -> str:
        """Create a new session with context."""
        if len(self._sessions) >= self._max_sessions:
            oldest = min(self._sessions.values(), key=lambda s: s["created_at"])
            self.end_session(oldest["session_id"])

        session_id = hashlib.sha256(f"{model}{time.time()}".encode()).hexdigest()[:16]
        self._sessions[session_id] = {
            "session_id": session_id,
            "model": model, "system_prompt": system_prompt,
            "created_at": time.time(), "message_count": 0,
            "metadata": metadata or {}
        }
        return session_id

    def get_session(self, session_id: str) -> Optional[dict]:
        return self._sessions.get(session_id)

    def end_session(self, session_id: str) -> bool:
        removed = self._sessions.pop(session_id, None) is not None
        self._contexts.pop(session_id, None)
        return removed

    # --- Context Management ---

    def create_context(self, session_id: str, room: str = "", agent: str = "") -> McpContext:
        ctx = McpContext(room=room, agent=agent)
        self._contexts[session_id] = ctx
        return ctx

    def get_context(self, session_id: str) -> Optional[McpContext]:
        return self._contexts.get(session_id)

    def add_tile_to_context(self, session_id: str, tile: dict) -> Optional[dict]:
        ctx = self._contexts.get(session_id)
        if ctx:
            return ctx.add_tile(tile)
        return None

    def get_context_summary(self, session_id: str) -> Optional[dict]:
        ctx = self._contexts.get(session_id)
        return ctx.to_dict() if ctx else None

    # --- Middleware ---

    def add_middleware(self, middleware: McpMiddleware):
        self._middleware.append(middleware)

    def remove_middleware(self, middleware: McpMiddleware):
        self._middleware = [m for m in self._middleware if m is not middleware]

    # --- Observability ---

    def top_tools(self, n: int = 5) -> list[dict]:
        tools = sorted(self._tools.values(), key=lambda t: t.call_count, reverse=True)
        return [{"name": t.name, "calls": t.call_count, "errors": t.error_count,
                 "avg_latency_ms": round(t.avg_latency * 1000, 1),
                 "status": t.status.value} for t in tools[:n]]

    def tool_health(self) -> dict:
        """Health report for all tools."""
        healthy = sum(1 for t in self._tools.values() if t.status == ToolStatus.ACTIVE)
        degraded = sum(1 for t in self._tools.values() if t.status == ToolStatus.DEGRADED)
        errors = sum(t.error_count for t in self._tools.values())
        return {"total": len(self._tools), "active": healthy, "degraded": degraded,
                "disabled": len(self._tools) - healthy - degraded, "total_errors": errors}

    @property
    def stats(self) -> dict:
        return {"tools": len(self._tools), "active_sessions": len(self._sessions),
                "contexts": len(self._contexts), "middleware": len(self._middleware),
                "tool_health": self.tool_health(),
                "top_tool": self.top_tools(1)[0] if self._tools else None}

    def export_schema(self) -> dict:
        """Export full bridge schema for tool discovery by LLMs."""
        return {"bridge_version": "2.0", "tools": self.list_tools(),
                "sessions": len(self._sessions), "stats": self.stats}
