"""MCP (Model Context Protocol) server router for Mem0.

Exposes 9 tools via MCP SDK 2.0 decorators that wrap the underlying REST API.
Mounted at /mcp in main.py via app.mount().

Auth: AuthMiddleware extracts credentials from request headers before each tool call
and stashes the verified User in a contextvar so tool implementations can read it.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from contextvars import ContextVar
from typing import Any

from contextlib import asynccontextmanager

from fastapi import HTTPException
from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import CallToolResult, Tool
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from auth import AUTH_DISABLED, api_key_header, auth_scheme, verify_auth
from server_state import get_memory_instance

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Auth context — carries the verified user from middleware into tool handlers
# ---------------------------------------------------------------------------

_auth_user_var: ContextVar[Any | None] = ContextVar("_auth_user_var", default=None)


def get_current_auth_user():
    """The User for the current request, set by AuthMiddleware."""
    return _auth_user_var.get()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TYPE_TO_FIELD = {"user": "user_id", "agent": "agent_id", "run": "run_id"}


def _error_result(message: str) -> CallToolResult:
    return CallToolResult(
        isError=True,
        content=[{"type": "text", "text": json.dumps({"error": message})}],
    )


def _ok_result(data: Any) -> CallToolResult:
    return CallToolResult(
        content=[{"type": "text", "text": json.dumps(data, default=str)}]
    )


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

# The decorated tool functions are registered on this server instance.
mcp_server = MCPServer(
    name="mem0-mcp",
    description="Mem0 memory server — add, search, and manage AI agent memories.",
    version="1.0.0",
)


@mcp_server.tool(
    name="add_memory",
    description="Store new memories from a list of messages.",
    annotations={"readOnly": False, "destructive": False},
)
async def add_memory(
    messages: list[dict[str, str]],
    user_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    expiration_date: str | None = None,
    infer: bool | None = None,
) -> CallToolResult:
    """Store new memories from a list of messages."""
    if not messages:
        return _error_result("messages is required and must be non-empty.")
    if not any([user_id, agent_id, run_id]):
        return _error_result("At least one identifier (user_id, agent_id, run_id) is required.")
    params: dict[str, Any] = {
        k: v for k, v in {
            "user_id": user_id,
            "agent_id": agent_id,
            "run_id": run_id,
            "metadata": metadata,
            "expiration_date": expiration_date,
            "infer": infer,
        }.items() if v is not None
    }
    try:
        return _ok_result(get_memory_instance().add(messages=messages, **params))
    except Exception as e:
        return _error_result(str(e))


@mcp_server.tool(
    name="search_memories",
    description="Search memories using natural language query.",
    annotations={"readOnly": True, "destructive": False},
)
async def search_memories(
    query: str,
    user_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
    top_k: int | None = None,
    threshold: float | None = None,
    filters: dict[str, Any] | None = None,
) -> CallToolResult:
    """Search memories using natural language query."""
    if not query:
        return _error_result("query is required.")
    filter_dict: dict[str, Any] = dict(filters) if filters else {}
    for key, val in [("user_id", user_id), ("agent_id", agent_id), ("run_id", run_id)]:
        if val:
            filter_dict[key] = val
    params: dict[str, Any] = {"filters": filter_dict}
    if top_k is not None:
        params["top_k"] = top_k
    if threshold is not None:
        params["threshold"] = threshold
    try:
        return _ok_result(get_memory_instance().search(query=query, **params))
    except Exception as e:
        return _error_result(str(e))


@mcp_server.tool(
    name="get_memories",
    description="List memories, optionally filtered by user/agent/run.",
    annotations={"readOnly": True, "destructive": False},
)
async def get_memories(
    user_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
    top_k: int | None = None,
    show_expired: bool | None = None,
) -> CallToolResult:
    """List memories, optionally filtered by user/agent/run."""
    filter_dict = {k: v for k, v in [("user_id", user_id), ("agent_id", agent_id), ("run_id", run_id)] if v}
    kwargs: dict[str, Any] = {"filters": filter_dict} if filter_dict else {}
    if top_k is not None:
        kwargs["top_k"] = top_k
    if show_expired is not None:
        kwargs["show_expired"] = show_expired
    try:
        return _ok_result(get_memory_instance().get_all(**kwargs))
    except Exception as e:
        return _error_result(str(e))


@mcp_server.tool(
    name="get_memory",
    description="Retrieve a specific memory by ID.",
    annotations={"readOnly": True, "destructive": False},
)
async def get_memory(memory_id: str) -> CallToolResult:
    """Retrieve a specific memory by ID."""
    if not memory_id:
        return _error_result("memory_id is required.")
    try:
        return _ok_result(get_memory_instance().get(memory_id))
    except Exception as e:
        return _error_result(str(e))


@mcp_server.tool(
    name="update_memory",
    description="Update an existing memory's text and/or metadata.",
    annotations={"readOnly": False, "destructive": False},
)
async def update_memory(
    memory_id: str,
    text: str | None = None,
    metadata: dict[str, Any] | None = None,
    expiration_date: str | None = None,
) -> CallToolResult:
    """Update an existing memory's text and/or metadata."""
    if not memory_id:
        return _error_result("memory_id is required.")
    params: dict[str, Any] = {"memory_id": memory_id}
    if text is not None:
        params["data"] = text
    if metadata is not None:
        params["metadata"] = metadata
    if expiration_date is not None:
        params["expiration_date"] = expiration_date
    try:
        return _ok_result(get_memory_instance().update(**params))
    except Exception as e:
        return _error_result(str(e))


@mcp_server.tool(
    name="delete_memory",
    description="Delete a specific memory by ID.",
    annotations={"readOnly": False, "destructive": True},
)
async def delete_memory(memory_id: str) -> CallToolResult:
    """Delete a specific memory by ID."""
    if not memory_id:
        return _error_result("memory_id is required.")
    try:
        get_memory_instance().delete(memory_id=memory_id)
        return _ok_result({"message": "Memory deleted successfully"})
    except Exception as e:
        return _error_result(str(e))


@mcp_server.tool(
    name="delete_all_memories",
    description="Delete all memories for a given user/agent/run. Requires admin auth.",
    annotations={"readOnly": False, "destructive": True},
)
async def delete_all_memories(
    user_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
) -> CallToolResult:
    """Delete all memories for a given user/agent/run. Requires admin auth."""
    if not any([user_id, agent_id, run_id]):
        return _error_result("At least one identifier (user_id, agent_id, run_id) is required.")
    # Admin-only. AUTH_DISABLED callers are treated as admin (dev mode), mirroring auth.require_admin.
    if not AUTH_DISABLED:
        user = get_current_auth_user()
        if user is None or getattr(user, "role", None) != "admin":
            return _error_result("Admin role required.")
    params = {k: v for k, v in [("user_id", user_id), ("agent_id", agent_id), ("run_id", run_id)] if v}
    try:
        get_memory_instance().delete_all(**params)
        return _ok_result({"message": "All relevant memories deleted"})
    except Exception as e:
        return _error_result(str(e))


@mcp_server.tool(
    name="delete_entities",
    description="Delete all memories for an entity type and ID (user/agent/run).",
    annotations={"readOnly": False, "destructive": True},
)
async def delete_entities(entity_type: str, entity_id: str) -> CallToolResult:
    """Delete all memories for an entity type and ID."""
    if not entity_type or not entity_id:
        return _error_result("entity_type and entity_id are required.")
    field = _TYPE_TO_FIELD.get(entity_type)
    if not field:
        return _error_result(f"Invalid entity_type '{entity_type}'. Must be one of: user, agent, run.")
    try:
        get_memory_instance().delete_all(**{field: entity_id})
        return _ok_result({"message": f"Entity {entity_type}/{entity_id} deleted"})
    except Exception as e:
        return _error_result(str(e))


@mcp_server.tool(
    name="list_entities",
    description="List all entities (users, agents, runs) with memory counts.",
    annotations={"readOnly": True, "destructive": False},
)
async def list_entities() -> CallToolResult:
    """List all entities (users, agents, runs) with memory counts."""
    try:
        result = get_memory_instance().vector_store.list(top_k=10_000)
        rows = result[0] if result and isinstance(result, list) and isinstance(result[0], list) else (result or [])
        buckets: dict[tuple, dict] = defaultdict(
            lambda: {"total_memories": 0, "created_at": None, "updated_at": None}
        )
        for row in rows:
            payload = getattr(row, "payload", None) or {}
            created = payload.get("created_at")
            updated = payload.get("updated_at")
            for et, field in _TYPE_TO_FIELD.items():
                val = payload.get(field)
                if not val:
                    continue
                bucket = buckets[(et, str(val))]
                bucket["total_memories"] += 1
                if created and (bucket["created_at"] is None or created < bucket["created_at"]):
                    bucket["created_at"] = created
                if updated and (bucket["updated_at"] is None or updated > bucket["updated_at"]):
                    bucket["updated_at"] = updated
        entities = [
            {"id": entity_id, "type": entity_type, **data}
            for (entity_type, entity_id), data in sorted(
                buckets.items(), key=lambda item: (item[0][0], item[0][1])
            )
        ]
        return _ok_result({"entities": entities})
    except Exception as e:
        return _error_result(str(e))


# ---------------------------------------------------------------------------
# Starlette app factory
# ---------------------------------------------------------------------------

# The raw MCP Starlette app, set by create_mcp_starlette_app(). Its lifespan
# must be run by the host application — see mcp_lifespan.
_mcp_asgi_app = None


@asynccontextmanager
async def mcp_lifespan(app):
    """Run the MCP session manager within the host app's lifespan.

    Starlette never forwards the lifespan scope to mounted sub-apps, so the
    StreamableHTTPSessionManager inside the MCP app is never initialized on its
    own and every request fails with "Task group is not initialized".
    """
    if _mcp_asgi_app is not None:
        async with _mcp_asgi_app.router.lifespan_context(_mcp_asgi_app):
            yield
    else:
        yield


def create_mcp_starlette_app():
    """
    Build and return the Starlette ASGI app for the MCP server.

    AuthMiddleware wraps the raw MCP Starlette app so every request
    (including streaming POST /mcp calls) authenticates before tool execution.
    The verified User is stored in a contextvar so tool handlers can read it.
    """
    global _mcp_asgi_app

    mcp_app = mcp_server.streamable_http_app(
        # main.py already mounts this app at /mcp; a nested "/mcp" here would
        # double the prefix and expose the endpoint at /mcp/mcp.
        streamable_http_path="/",
        json_response=False,
        # The SDK auto-enables localhost-only Host-header checks when no host is
        # given, which would 403 every client arriving via the LAN/docker
        # address. Authentication is enforced by AuthMiddleware instead.
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )

    class AuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            if AUTH_DISABLED:
                user = None
            else:
                try:
                    # Middleware runs outside FastAPI's dependency injection, so the
                    # Depends() defaults on verify_auth are never resolved. Extract the
                    # credentials here and pass them explicitly.
                    credentials = await auth_scheme(request)
                    x_api_key = await api_key_header(request)
                    user = await verify_auth(request, credentials=credentials, x_api_key=x_api_key)
                except HTTPException:
                    user = None

            if user is None and not AUTH_DISABLED:
                return JSONResponse(
                    status_code=401,
                    content={"error": "Authentication required."},
                )

            token = _auth_user_var.set(user)
            try:
                return await call_next(request)
            finally:
                _auth_user_var.reset(token)

    _mcp_asgi_app = mcp_app
    return AuthMiddleware(mcp_app)
