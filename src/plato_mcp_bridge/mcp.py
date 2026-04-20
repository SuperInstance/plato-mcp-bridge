"""Model Context Protocol bridge — connect LLM tools to PLATO rooms."""
import time
import json
from dataclasses import dataclass, field
from typing import Callable, Optional
from collections import defaultdict

@dataclass
class McpTool:
    name: str
    description: str
    handler: str  # reference to callable
    parameters: dict = field(default_factory=dict)
    enabled: bool = True
    call_count: int = 0
    avg_latency: float = 0.0
    total_latency: float = 0.0

@dataclass
class McpContext:
    room: str = ""
    agent: str = ""
    tiles: list[dict] = field(default_factory=list)
    variables: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)

class McpBridge:
    def __init__(self):
        self._tools: dict[str, McpTool] = {}
        self._contexts: dict[str, McpContext] = {}
        self._call_log: list[dict] = []
        self._sessions: dict[str, dict] = {}

    def register_tool(self, name: str, description: str, parameters: dict = None) -> McpTool:
        tool = McpTool(name=name, description=description, parameters=parameters or {})
        self._tools[name] = tool
        return tool

    def unregister_tool(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    def list_tools(self, room: str = "") -> list[dict]:
        return [{"name": t.name, "description": t.description, "parameters": t.parameters,
                 "enabled": t.enabled, "call_count": t.call_count}
                for t in self._tools.values() if t.enabled]

    def call_tool(self, name: str, context: McpContext = None, **kwargs) -> dict:
        start = time.time()
        tool = self._tools.get(name)
        if not tool or not tool.enabled:
            return {"error": f"Tool '{name}' not found or disabled", "status": "error"}
        tool.call_count += 1
        latency = time.time() - start
        tool.total_latency += latency
        tool.avg_latency = tool.total_latency / tool.call_count
        result = {"tool": name, "status": "ok", "latency_ms": round(latency * 1000, 1),
                 "context_room": context.room if context else "",
                 "kwargs": kwargs, "timestamp": time.time()}
        self._call_log.append(result)
        if len(self._call_log) > 500:
            self._call_log = self._call_log[-500:]
        return result

    def create_context(self, session_id: str, room: str = "", agent: str = "") -> McpContext:
        ctx = McpContext(room=room, agent=agent)
        self._contexts[session_id] = ctx
        return ctx

    def get_context(self, session_id: str) -> Optional[McpContext]:
        return self._contexts.get(session_id)

    def add_tile_to_context(self, session_id: str, tile: dict):
        ctx = self._contexts.get(session_id)
        if ctx:
            ctx.tiles.append(tile)

    def create_session(self, model: str = "default", system_prompt: str = "") -> str:
        import hashlib
        session_id = hashlib.sha256(f"{model}{time.time()}".encode()).hexdigest()[:16]
        self._sessions[session_id] = {"model": model, "system_prompt": system_prompt,
                                        "created_at": time.time(), "message_count": 0}
        return session_id

    def end_session(self, session_id: str):
        self._sessions.pop(session_id, None)
        self._contexts.pop(session_id, None)

    def top_tools(self, n: int = 5) -> list[dict]:
        tools = sorted(self._tools.values(), key=lambda t: t.call_count, reverse=True)
        return [{"name": t.name, "calls": t.call_count, "avg_latency_ms": round(t.avg_latency * 1000, 1)}
                for t in tools[:n]]

    @property
    def stats(self) -> dict:
        return {"tools": len(self._tools), "active_sessions": len(self._sessions),
                "contexts": len(self._contexts), "total_calls": len(self._call_log),
                "top_tool": self.top_tools(1)[0]["name"] if self._tools else "none"}
