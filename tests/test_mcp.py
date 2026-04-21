"""Tests for plato-mcp-bridge."""
import time
import pytest
from plato_mcp_bridge import McpBridge, McpTool, McpContext, LoggingMiddleware, ToolStatus


def test_register_and_list():
    b = McpBridge()
    b.register_tool("echo", "Echo input", parameters={"input": {"type": "string"}})
    tools = b.list_tools()
    assert len(tools) == 1
    assert tools[0]["name"] == "echo"

def test_call_tool_not_found():
    b = McpBridge()
    result = b.call_tool("nonexistent")
    assert result["status"] == "not_found"

def test_call_tool_with_handler():
    b = McpBridge()
    b.register_tool("double", "Double a number", handler=lambda n, **kw: {"result": n * 2})
    r = b.call_tool("double", n=21)
    assert r["result"]["result"] == 42
    assert r["status"] == "ok"
    assert r["latency_ms"] >= 0

def test_call_tool_no_handler():
    b = McpBridge()
    b.register_tool("noop", "No-op")
    r = b.call_tool("noop", x=1)
    assert r["status"] == "ok"
    assert "kwargs" in r["result"]

def test_handler_error():
    b = McpBridge()
    def bad_handler(**kw):
        raise ValueError("boom")
    b.register_tool("fail", "Always fails", handler=bad_handler)
    r = b.call_tool("fail")
    assert r["status"] == "error"
    assert "boom" in r["error"]

def test_disable_tool():
    b = McpBridge()
    b.register_tool("t", "test")
    b.disable_tool("t")
    r = b.call_tool("t")
    assert r["status"] == "disabled"
    b.enable_tool("t")
    r = b.call_tool("t")
    assert r["status"] == "ok"

def test_rate_limit():
    b = McpBridge()
    b.register_tool("limited", "Rate limited", rate_limit=2)
    assert b.call_tool("limited")["status"] == "ok"
    assert b.call_tool("limited")["status"] == "ok"
    assert b.call_tool("limited")["status"] == "rate_limited"

def test_session_lifecycle():
    b = McpBridge()
    sid = b.create_session(model="gpt-4")
    assert b.get_session(sid) is not None
    assert b.get_session(sid)["model"] == "gpt-4"
    assert b.end_session(sid) is True
    assert b.get_session(sid) is None

def test_context_management():
    b = McpBridge()
    sid = b.create_session()
    ctx = b.create_context(sid, room="forge", agent="forgemaster")
    assert ctx.room == "forge"
    b.add_tile_to_context(sid, {"text": "hello", "type": "message"})
    summary = b.get_context_summary(sid)
    assert summary["tile_count"] == 1
    assert summary["room"] == "forge"

def test_context_tile_overflow():
    ctx = McpContext(max_tiles=5)
    for i in range(8):
        ctx.add_tile({"id": i})
    assert len(ctx.tiles) == 5
    assert ctx.tiles[-1]["id"] == 7

def test_context_variables():
    ctx = McpContext()
    ctx.set_variable("learning_rate", 0.001)
    assert ctx.get_variable("learning_rate") == 0.001
    assert ctx.get_variable("missing", "default") == "default"

def test_context_history():
    ctx = McpContext(max_history=3)
    for i in range(5):
        ctx.add_history_entry("tool", f"call {i}", tool_name="t")
    assert len(ctx.history) == 3
    assert ctx.history[-1]["tool"] == "t"

def test_batch_calls():
    b = McpBridge()
    b.register_tool("add", "Add", handler=lambda a, b=0, **kw: a + b)
    results = b.call_tool_batch([
        {"tool": "add", "kwargs": {"a": 1, "b": 2}},
        {"tool": "add", "kwargs": {"a": 10, "b": 20}},
    ])
    assert results[0]["result"] == 3
    assert results[1]["result"] == 30

def test_middleware_logging():
    b = McpBridge()
    log_mw = LoggingMiddleware()
    b.add_middleware(log_mw)
    b.register_tool("ping", "Ping", handler=lambda **kw: "pong")
    b.call_tool("ping")
    log = log_mw.get_log()
    assert len(log) == 1
    assert log[0]["tool"] == "ping"

def test_discover_tools():
    b = McpBridge()
    b.register_tool("tile_search", "Search tiles")
    b.register_tool("room_create", "Create room")
    results = b.discover_tools("tile")
    assert len(results) == 1
    assert results[0]["name"] == "tile_search"

def test_top_tools():
    b = McpBridge()
    b.register_tool("popular", "Popular", handler=lambda **kw: None)
    b.register_tool("unpopular", "Unpopular", handler=lambda **kw: None)
    b.call_tool("popular")
    b.call_tool("popular")
    b.call_tool("unpopular")
    top = b.top_tools(2)
    assert top[0]["name"] == "popular"
    assert top[0]["calls"] == 2

def test_tool_health():
    b = McpBridge()
    b.register_tool("ok", "OK", handler=lambda **kw: None)
    b.register_tool("bad", "Bad", handler=lambda **kw: (_ for _ in ()).throw(RuntimeError("err")))
    b.call_tool("ok")
    b.call_tool("bad")
    health = b.tool_health()
    assert health["active"] == 2
    assert health["total_errors"] == 1

def test_export_schema():
    b = McpBridge()
    b.register_tool("test", "Test tool", parameters={"x": {"type": "integer"}})
    schema = b.export_schema()
    assert schema["bridge_version"] == "2.0"
    assert len(schema["tools"]) == 1

def test_unregister_tool():
    b = McpBridge()
    b.register_tool("temp", "Temporary")
    assert b.unregister_tool("temp") is True
    assert b.unregister_tool("temp") is False

def test_session_max_eviction():
    b = McpBridge(default_max_sessions=2)
    s1 = b.create_session()
    s2 = b.create_session()
    s3 = b.create_session()
    assert b.get_session(s1) is None
    assert b.get_session(s3) is not None
