#!/usr/bin/env python3
"""scripts/mcp_server.py — Capibara Legal MCP Server (Model Context Protocol 2024-11-05)

Wraps ``scripts/tool_executor.py`` (5 legal tools) and ``scripts/rag_retriever.py``
(RAGRetriever) as a standard MCP server so that Claude Code, Claude Desktop, and
any other MCP-compatible client can call them directly.

Transports
----------
stdio (default)
    JSON-RPC 2.0 messages delimited by newlines on stdin/stdout.
    This is the standard transport for Claude Code.

HTTP (``--http --port PORT``)
    Minimal HTTP server; POST JSON-RPC bodies to ``/mcp``.
    Useful for manual testing with curl/httpie.

Usage
-----
    # stdio mode — wire up to Claude Code
    python scripts/mcp_server.py --rag-index data/rag_index/

    # HTTP mode — manual testing
    python scripts/mcp_server.py --http --port 3000 --rag-index data/rag_index/

    # Quick sanity check (no RAG needed)
    echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | python scripts/mcp_server.py

MCP config snippet (printed to stderr on startup)
--------------------------------------------------
Add to .claude/settings.json mcpServers:
{
  "capibara-legal": {
    "command": "python",
    "args": ["scripts/mcp_server.py", "--rag-index", "data/rag_index/"]
  }
}

Exposed tools (from tool_executor.ToolRegistry)
-------------------------------------------------
  search_boe        — Full-text search in Boletín Oficial del Estado
  search_cendoj     — Search CENDOJ jurisprudence database
  search_web        — General web search with legal focus
  get_article       — Retrieve a specific law article by reference
  calculate_plazo   — Calculate procedural deadlines (plazos)

Exposed resources
-----------------
  legal://corpus    — RAG retrieval over the Spanish legal corpus
                      Pass ``?q=<query>`` in the URI to retrieve chunks.

Notes
-----
- All logging goes to stderr; stdout is reserved for MCP JSON in stdio mode.
- No external dependencies — pure stdlib (Python 3.9+).
- tool_executor and rag_retriever are lazy-imported so the server starts fast
  even when those modules have heavy dependencies.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

# ---------------------------------------------------------------------------
# Logging — stderr only so stdout stays clean for JSON-RPC in stdio mode
# ---------------------------------------------------------------------------

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("capibara.mcp")

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "capibara-legal"
SERVER_VERSION = "1.0.0"

# JSON-RPC error codes
_ERR_PARSE_ERROR     = -32700
_ERR_INVALID_REQUEST = -32600
_ERR_METHOD_NOT_FOUND = -32601
_ERR_INVALID_PARAMS  = -32602
_ERR_INTERNAL        = -32603

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

def _scripts_dir() -> str:
    """Return the absolute path to the scripts/ directory."""
    return str(Path(__file__).parent.resolve())


def _ensure_scripts_on_path() -> None:
    scripts = _scripts_dir()
    if scripts not in sys.path:
        sys.path.insert(0, scripts)


# ---------------------------------------------------------------------------
# ToolDispatcher
# ---------------------------------------------------------------------------

class ToolDispatcher:
    """Lazy-loads tool_executor and dispatches MCP tool calls.

    The import is deferred so the MCP server can start even when
    tool_executor's optional dependencies (e.g. httpx, lxml) are not
    installed — only the first tool call will trigger the import error.
    """

    # These match the tool_executor.ToolRegistry exactly.
    TOOL_SCHEMAS: List[Dict[str, Any]] = [
        {
            "name": "search_boe",
            "description": (
                "Busca en el Boletín Oficial del Estado (BOE) legislación española, "
                "reglamentos, resoluciones y anuncios oficiales."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Texto de búsqueda (p.ej. 'artículo 248 código penal')",
                    },
                    "year_from": {
                        "type": "integer",
                        "description": "Año mínimo de publicación (opcional, p.ej. 2000)",
                    },
                    "year_to": {
                        "type": "integer",
                        "description": "Año máximo de publicación (opcional)",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Número máximo de resultados (por defecto 10)",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "search_cendoj",
            "description": (
                "Busca jurisprudencia en el Centro de Documentación Judicial (CENDOJ): "
                "sentencias del Tribunal Supremo, Audiencias Provinciales, etc."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Texto de búsqueda jurisprudencial",
                    },
                    "tribunal": {
                        "type": "string",
                        "description": "Filtro de tribunal (p.ej. 'Tribunal Supremo')",
                    },
                    "sala": {
                        "type": "string",
                        "description": "Sala o sección (p.ej. 'Sala de lo Civil')",
                    },
                    "year_from": {
                        "type": "integer",
                        "description": "Año mínimo de la sentencia",
                    },
                    "year_to": {
                        "type": "integer",
                        "description": "Año máximo de la sentencia",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Número máximo de resultados (por defecto 10)",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "search_web",
            "description": (
                "Búsqueda web general con enfoque jurídico-legal español. "
                "Útil para doctrina, comentarios de artículos, noticias legislativas."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Consulta de búsqueda web",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Número máximo de resultados (por defecto 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "get_article",
            "description": (
                "Recupera el texto completo de un artículo concreto de la legislación "
                "española (Código Civil, Código Penal, LEC, etc.)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "law": {
                        "type": "string",
                        "description": (
                            "Nombre o identificador de la ley "
                            "(p.ej. 'codigo-civil', 'codigo-penal', 'LEC')"
                        ),
                    },
                    "article": {
                        "type": "string",
                        "description": "Número del artículo (p.ej. '248', '1089', '24')",
                    },
                },
                "required": ["law", "article"],
            },
        },
        {
            "name": "calculate_plazo",
            "description": (
                "Calcula plazos procesales conforme al calendario oficial español, "
                "teniendo en cuenta días hábiles, festivos nacionales y autonómicos."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "fecha_inicio": {
                        "type": "string",
                        "description": "Fecha de inicio en formato YYYY-MM-DD",
                    },
                    "dias": {
                        "type": "integer",
                        "description": "Número de días del plazo",
                    },
                    "tipo": {
                        "type": "string",
                        "enum": ["habiles", "naturales"],
                        "description": "Tipo de días: 'habiles' (hábiles) o 'naturales'",
                        "default": "habiles",
                    },
                    "comunidad": {
                        "type": "string",
                        "description": (
                            "Código de comunidad autónoma para festivos autonómicos "
                            "(p.ej. 'CAT', 'AND', 'MAD'). Opcional."
                        ),
                    },
                },
                "required": ["fecha_inicio", "dias"],
            },
        },
    ]

    def __init__(self) -> None:
        self._registry: Any = None
        self._lock = threading.Lock()

    def _get_registry(self) -> Any:
        """Return the ToolRegistry, importing tool_executor on first call."""
        if self._registry is not None:
            return self._registry
        with self._lock:
            if self._registry is None:
                _ensure_scripts_on_path()
                try:
                    import tool_executor  # type: ignore[import]
                    self._registry = tool_executor.ToolRegistry()
                    logger.info("tool_executor loaded successfully")
                except ImportError as exc:
                    raise RuntimeError(
                        f"Cannot import tool_executor: {exc}. "
                        "Make sure scripts/tool_executor.py exists and its "
                        "dependencies are installed."
                    ) from exc
        return self._registry

    def call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch a tool call. Returns MCP content list."""
        try:
            registry = self._get_registry()
            result = registry.call(name, arguments)
            # tool_executor.ToolRegistry.call() should return a string or
            # something str()-able; we always wrap it as MCP text content.
            return {
                "content": [{"type": "text", "text": str(result)}],
                "isError": False,
            }
        except KeyError:
            return {
                "content": [
                    {"type": "text", "text": f"Unknown tool: {name!r}"}
                ],
                "isError": True,
            }
        except Exception as exc:
            logger.warning("Tool %r raised %s: %s", name, type(exc).__name__, exc)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Tool error ({type(exc).__name__}): {exc}",
                    }
                ],
                "isError": True,
            }


# ---------------------------------------------------------------------------
# RAGResource
# ---------------------------------------------------------------------------

class RAGResource:
    """Exposes the legal RAG corpus as an MCP resource.

    Only activated when ``--rag-index`` is provided and the path exists.
    If RAG is not configured, ``resources/list`` returns an empty list.
    """

    RESOURCE_URI = "legal://corpus"
    RESOURCE_DESCRIPTOR = {
        "uri": RESOURCE_URI,
        "name": "Corpus Legal Español",
        "description": (
            "BOE, sentencias del Tribunal Constitucional y Tribunal Supremo, "
            "legislación española consolidada. Consulta semántica vía RAG."
        ),
        "mimeType": "text/plain",
    }

    def __init__(self, index_path: Optional[str] = None) -> None:
        self._index_path = index_path
        self._retriever: Any = None
        self._available: Optional[bool] = None
        self._lock = threading.Lock()

    @property
    def is_configured(self) -> bool:
        """True if an index path was provided and it exists on disk."""
        if self._index_path is None:
            return False
        return Path(self._index_path).exists()

    def _get_retriever(self) -> Any:
        """Return the RAGRetriever, importing rag_retriever on first call."""
        if self._retriever is not None:
            return self._retriever
        with self._lock:
            if self._retriever is None:
                _ensure_scripts_on_path()
                try:
                    import rag_retriever  # type: ignore[import]
                    self._retriever = rag_retriever.RAGRetriever(self._index_path)
                    logger.info("RAGRetriever loaded from %s", self._index_path)
                except ImportError as exc:
                    raise RuntimeError(
                        f"Cannot import rag_retriever: {exc}. "
                        "Make sure scripts/rag_retriever.py exists."
                    ) from exc
        return self._retriever

    def list_resources(self) -> List[Dict[str, Any]]:
        if not self.is_configured:
            return []
        return [self.RESOURCE_DESCRIPTOR]

    def read(self, uri: str) -> Dict[str, Any]:
        """Handle ``resources/read`` for ``legal://corpus?q=...``."""
        if not self.is_configured:
            return {
                "error": "RAG index not configured. "
                "Start the server with --rag-index <path>."
            }
        parsed = urlparse(uri)
        qs = parse_qs(parsed.query)
        query_parts = qs.get("q", [])
        query = " ".join(query_parts).strip()

        try:
            retriever = self._get_retriever()
            if query:
                chunks = retriever.retrieve(query)
            else:
                # No query — return a short description of the corpus
                chunks = retriever.describe() if hasattr(retriever, "describe") else (
                    "Corpus Legal Español — proporciona ?q=<consulta> para recuperar fragmentos."
                )
            text = chunks if isinstance(chunks, str) else "\n\n---\n\n".join(chunks)
            return {
                "contents": [
                    {
                        "uri": self.RESOURCE_URI,
                        "mimeType": "text/plain",
                        "text": text,
                    }
                ]
            }
        except Exception as exc:
            logger.warning("RAG retrieval failed: %s", exc)
            return {
                "contents": [
                    {
                        "uri": self.RESOURCE_URI,
                        "mimeType": "text/plain",
                        "text": f"Error de recuperación RAG: {exc}",
                    }
                ]
            }


# ---------------------------------------------------------------------------
# JSON-RPC helpers
# ---------------------------------------------------------------------------

def _ok(req_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _err(req_id: Any, code: int, message: str, data: Any = None) -> Dict[str, Any]:
    error: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": error}


def _encode(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


# ---------------------------------------------------------------------------
# MCP message router
# ---------------------------------------------------------------------------

class MCPRouter:
    """Routes JSON-RPC messages to the appropriate handler.

    Stateless beyond the tool dispatcher and RAG resource instances, so a
    single router can be safely shared across threads (HTTP mode).
    """

    def __init__(
        self,
        dispatcher: ToolDispatcher,
        rag: RAGResource,
    ) -> None:
        self._dispatcher = dispatcher
        self._rag = rag

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def handle(self, raw: str) -> Optional[str]:
        """Process one raw JSON string.  Returns a JSON response string,
        or None for notifications (no response expected)."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError as exc:
            return _encode(_err(None, _ERR_PARSE_ERROR, f"Parse error: {exc}"))

        if not isinstance(msg, dict):
            return _encode(
                _err(None, _ERR_INVALID_REQUEST, "Request must be a JSON object")
            )

        req_id  = msg.get("id")      # None for notifications
        method  = msg.get("method", "")
        params  = msg.get("params") or {}

        # Notifications (no id) — process but return nothing
        if "id" not in msg:
            if method == "notifications/initialized":
                logger.debug("Client initialized")
            else:
                logger.debug("Unhandled notification: %s", method)
            return None

        # Dispatch
        try:
            result = self._dispatch(method, params)
        except Exception as exc:
            logger.error("Unhandled exception in %r: %s", method, exc, exc_info=True)
            return _encode(_err(req_id, _ERR_INTERNAL, f"Internal error: {exc}"))

        if result is None:
            # Method explicitly returned None → method not found
            return _encode(
                _err(req_id, _ERR_METHOD_NOT_FOUND, f"Method not found: {method!r}")
            )

        return _encode(_ok(req_id, result))

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def _dispatch(self, method: str, params: Any) -> Optional[Any]:
        """Return a result dict/list, or None if the method is unknown."""
        if method == "initialize":
            return self._handle_initialize(params)
        if method == "ping":
            return {}
        if method == "tools/list":
            return self._handle_tools_list()
        if method == "tools/call":
            return self._handle_tools_call(params)
        if method == "resources/list":
            return self._handle_resources_list()
        if method == "resources/read":
            return self._handle_resources_read(params)
        return None  # signals method not found

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _handle_initialize(self, params: Any) -> Dict[str, Any]:
        client_version = (params or {}).get("protocolVersion", "unknown")
        logger.info(
            "Client initialized — protocolVersion=%s", client_version
        )
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {
                "tools": {},
                "resources": {},
            },
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
            },
        }

    def _handle_tools_list(self) -> Dict[str, Any]:
        return {"tools": self._dispatcher.TOOL_SCHEMAS}

    def _handle_tools_call(self, params: Any) -> Dict[str, Any]:
        if not isinstance(params, dict):
            return {
                "content": [{"type": "text", "text": "Invalid params: expected object"}],
                "isError": True,
            }
        name      = params.get("name", "")
        arguments = params.get("arguments") or {}
        if not name:
            return {
                "content": [{"type": "text", "text": "Missing tool name"}],
                "isError": True,
            }
        logger.info("tools/call  name=%r  args=%s", name, list(arguments.keys()))
        return self._dispatcher.call(name, arguments)

    def _handle_resources_list(self) -> Dict[str, Any]:
        return {"resources": self._rag.list_resources()}

    def _handle_resources_read(self, params: Any) -> Dict[str, Any]:
        if not isinstance(params, dict) or "uri" not in params:
            return {
                "contents": [
                    {
                        "uri": "",
                        "mimeType": "text/plain",
                        "text": "Missing 'uri' parameter",
                    }
                ]
            }
        uri = params["uri"]
        logger.info("resources/read  uri=%r", uri)
        return self._rag.read(uri)


# ---------------------------------------------------------------------------
# Transport: stdio
# ---------------------------------------------------------------------------

def run_stdio(router: MCPRouter) -> None:
    """Read newline-delimited JSON from stdin; write responses to stdout."""
    logger.info("Capibara Legal MCP server started (stdio transport)")
    # Use binary mode to avoid platform encoding surprises, then decode/encode
    # with UTF-8 explicitly.
    stdin  = sys.stdin.buffer
    stdout = sys.stdout.buffer

    while True:
        try:
            line = stdin.readline()
        except KeyboardInterrupt:
            logger.info("Interrupted — shutting down")
            break

        if not line:
            # EOF — client disconnected
            logger.info("stdin EOF — shutting down")
            break

        raw = line.decode("utf-8", errors="replace").strip()
        if not raw:
            continue

        response = router.handle(raw)
        if response is not None:
            stdout.write((response + "\n").encode("utf-8"))
            stdout.flush()


# ---------------------------------------------------------------------------
# Transport: HTTP
# ---------------------------------------------------------------------------

def _make_http_handler(router: MCPRouter):
    """Return a BaseHTTPRequestHandler subclass closed over *router*."""

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # type: ignore[override]
            # Route access log to our logger rather than stdout
            logger.debug("HTTP %s", fmt % args)

        def do_POST(self) -> None:
            if self.path != "/mcp":
                self.send_error(404, "Not found — POST to /mcp")
                return

            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length).decode("utf-8", errors="replace")

            response = router.handle(body)
            if response is None:
                # Notification — return 204
                self.send_response(204)
                self.end_headers()
                return

            payload = response.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            if self.path == "/health":
                body = b'{"status":"ok","server":"capibara-legal"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404, "Not found")

    return _Handler


def run_http(router: MCPRouter, host: str, port: int) -> None:
    handler = _make_http_handler(router)
    server  = HTTPServer((host, port), handler)
    logger.info(
        "Capibara Legal MCP server started (HTTP transport) — http://%s:%d/mcp",
        host,
        port,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Interrupted — shutting down HTTP server")
    finally:
        server.server_close()


# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------

_MCP_CONFIG_SNIPPET = """\
Add to .claude/settings.json mcpServers:
{
  "capibara-legal": {
    "command": "python",
    "args": ["scripts/mcp_server.py", "--rag-index", "data/rag_index/"]
  }
}"""


def _print_startup_banner(rag_index: Optional[str], http: bool, port: int) -> None:
    logger.info("=" * 60)
    logger.info("Capibara Legal MCP Server  v%s", SERVER_VERSION)
    logger.info("Protocol : %s", PROTOCOL_VERSION)
    logger.info("Transport: %s", f"HTTP :{port}" if http else "stdio")
    logger.info(
        "RAG index: %s",
        rag_index if rag_index and Path(rag_index).exists() else "not configured",
    )
    logger.info("=" * 60)
    # Print config snippet to stderr so it appears for the user but does not
    # contaminate the JSON-RPC stream on stdout.
    for line in _MCP_CONFIG_SNIPPET.splitlines():
        sys.stderr.write(line + "\n")
    sys.stderr.write("\n")
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcp_server.py",
        description="Capibara Legal MCP server — wraps tool_executor and rag_retriever.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--rag-index",
        metavar="PATH",
        default=None,
        help="Path to a pre-built RAG index directory (optional). "
             "When omitted, resources/list returns an empty list.",
    )
    parser.add_argument(
        "--http",
        action="store_true",
        default=False,
        help="Use HTTP transport instead of stdio.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=3000,
        metavar="PORT",
        help="HTTP port (default: 3000, only used with --http).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        metavar="HOST",
        help="HTTP bind address (default: 127.0.0.1, only used with --http).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser


def main() -> None:
    parser  = _build_arg_parser()
    args    = parser.parse_args()

    # Apply log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    _print_startup_banner(args.rag_index, args.http, args.port)

    dispatcher = ToolDispatcher()
    rag        = RAGResource(args.rag_index)
    router     = MCPRouter(dispatcher, rag)

    if args.http:
        run_http(router, args.host, args.port)
    else:
        run_stdio(router)


if __name__ == "__main__":
    main()
