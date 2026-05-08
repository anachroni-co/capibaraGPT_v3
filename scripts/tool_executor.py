#!/usr/bin/env python3
"""Tool executor for Capibara Legal LLM — five legal research tools + ToolRegistry.

This module exposes five tools designed for Spanish legal research and wires them
into a ToolRegistry that understands the byte-level 0xFF (ÿ) delimiter protocol
used by the capibaraGPT byte-level tokeniser (vocab=512, IDs 0-255 = raw UTF-8
bytes, 256-511 reserved; byte 0xFF / ID=255 never appears in valid UTF-8 and is
therefore safe as an unambiguous delimiter).

Tool call syntax in the byte stream:
    ÿTOOL:{"name":"search_boe","query":"recurso de amparo"}ÿ  →  model request
    ÿRESULT:{"status":"ok","data":[...]}ÿ                     →  injected result

Tools
-----
  search_boe(query, year_from=None, max_results=5)
      BOE open-data full-text search.

  search_cendoj(query, max_results=5)
      CENDOJ public portal — Tribunal Supremo sentencias.

  search_web(query, max_results=5)
      DuckDuckGo web search with legal query refinement.  Tries the
      ``duckduckgo-search`` package first; falls back to HTML scraping.

  get_article(ley, articulo)
      Fetch the text of a specific article from a known Spanish law via BOE.

  calculate_plazo(fecha_inicio, dias, tipo="habiles")
      Compute a procedural deadline (días hábiles o naturales).

Usage
-----
    python scripts/tool_executor.py --tool search_boe --query "recurso de amparo"
    python scripts/tool_executor.py --tool get_article --ley codigo_penal --articulo 248
    python scripts/tool_executor.py --tool calculate_plazo --fecha 2025-03-01 --dias 20 --tipo habiles
    python scripts/tool_executor.py --tool search_web --query "sentencia tribunal supremo daños morales 2024"
    python scripts/tool_executor.py --tool search_cendoj --query "responsabilidad civil medica"
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from html.parser import HTMLParser
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tool_executor")

# ─── Protocol constants ───────────────────────────────────────────────────────

DELIM = b"\xff"          # byte 0xFF — invalid in UTF-8, safe delimiter
DELIM_STR = "\xff"       # same as a Python str character (latin-1 code point)

# ─── HTTP helpers ─────────────────────────────────────────────────────────────

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; CapibaraLegal/3.0; "
        "+https://github.com/capibara-legal)"
    ),
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "es-ES,es;q=0.9",
}


def _http_get(
    url: str,
    *,
    timeout: int = 15,
    retries: int = 3,
    headers: dict[str, str] | None = None,
    method: str = "GET",
    data: bytes | None = None,
) -> bytes | None:
    """HTTP GET/POST with exponential backoff.  Returns bytes or None on failure."""
    merged = {**_DEFAULT_HEADERS, **(headers or {})}
    if data is not None:
        merged.setdefault("Content-Type", "application/x-www-form-urlencoded")
    delay = 2.0
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=merged, method=method, data=data)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 410):
                logger.debug("HTTP %d for %s — giving up", exc.code, url)
                return None
            logger.debug("HTTP %d for %s (attempt %d/%d)", exc.code, url, attempt + 1, retries)
        except urllib.error.URLError as exc:
            logger.debug("URLError %s for %s (attempt %d/%d)", exc.reason, url, attempt + 1, retries)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Error fetching %s: %s (attempt %d/%d)", url, exc, attempt + 1, retries)
        if attempt < retries - 1:
            time.sleep(delay)
            delay *= 2
    return None


def _strip_html(html: str) -> str:
    """Remove HTML tags and collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"&#?\w+;", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


# ─── Tool 1: search_boe ───────────────────────────────────────────────────────

BOE_SEARCH_URL = "https://www.boe.es/datosabiertos/api/search/hitwords"
BOE_DOC_BASE   = "https://www.boe.es/diario_boe/txt.php?id={doc_id}"


def search_boe(
    query: str,
    year_from: int | None = None,
    max_results: int = 5,
) -> list[dict[str, str]]:
    """Search the BOE open-data API and return up to *max_results* documents.

    Parameters
    ----------
    query:       Full-text search query (Spanish).
    year_from:   Optional lower bound year (e.g. 2020).
    max_results: Maximum number of results to return (default 5).

    Returns a list of dicts: {id, titulo, fecha, url, resumen}.
    """
    params: dict[str, str] = {"q": query}
    if year_from is not None:
        params["fecha_desde"] = f"{year_from}0101"
    url = BOE_SEARCH_URL + "?" + urllib.parse.urlencode(params)
    logger.info("BOE search: %s", url)

    raw = _http_get(url, headers={"Accept": "application/json"})
    if raw is None:
        logger.warning("BOE API returned no data for query '%s'", query)
        return []

    try:
        payload = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        logger.warning("BOE API JSON parse error: %s", exc)
        return []

    # BOE open-data response may nest results under different keys depending
    # on endpoint version.  Try common paths gracefully.
    items = (
        payload.get("response", {}).get("docs")
        or payload.get("docs")
        or payload.get("results")
        or (payload if isinstance(payload, list) else [])
    )

    results: list[dict[str, str]] = []
    for item in items[:max_results]:
        doc_id = item.get("identificador") or item.get("id") or ""
        titulo = item.get("titulo") or item.get("title") or "Sin título"
        fecha  = item.get("fecha_publicacion") or item.get("fecha") or ""
        url_   = item.get("url_html") or (BOE_DOC_BASE.format(doc_id=doc_id) if doc_id else "")
        resumen_raw = (
            item.get("resumen")
            or item.get("texto")
            or item.get("snippet")
            or ""
        )
        resumen = _strip_html(str(resumen_raw))[:300]
        results.append({
            "id":      doc_id,
            "titulo":  _strip_html(str(titulo)),
            "fecha":   str(fecha),
            "url":     url_,
            "resumen": resumen,
        })

    logger.info("BOE search returned %d result(s)", len(results))
    return results


# ─── Tool 2: search_cendoj ────────────────────────────────────────────────────

CENDOJ_URL = "https://www.poderjudicial.es/search/Search/action"
CENDOJ_FALLBACK = "https://www.poderjudicial.es/search/AN/openinterface.action"


class _CendojParser(HTMLParser):
    """Minimal HTML parser that extracts <div class="verdictresult"> blocks."""

    def __init__(self) -> None:
        super().__init__()
        self._in_result: int = 0          # nesting depth inside target div
        self._current: list[str] = []
        self.snippets: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_dict = dict(attrs)
        css = attr_dict.get("class") or ""
        if tag == "div" and "verdictresult" in css:
            self._in_result += 1
        elif self._in_result > 0 and tag == "div":
            self._in_result += 1

    def handle_endtag(self, tag: str) -> None:
        if self._in_result > 0 and tag == "div":
            self._in_result -= 1
            if self._in_result == 0 and self._current:
                self.snippets.append(" ".join(self._current).strip())
                self._current = []

    def handle_data(self, data: str) -> None:
        if self._in_result > 0:
            stripped = data.strip()
            if stripped:
                self._current.append(stripped)


def search_cendoj(
    query: str,
    max_results: int = 5,
) -> list[dict[str, str]]:
    """Search CENDOJ for Tribunal Supremo sentencias.

    Parameters
    ----------
    query:       Free-text query (Spanish legal terms).
    max_results: Maximum number of results to return (default 5).

    Returns a list of dicts: {titulo, tribunal, fecha, resumen}.
    Falls back to returning the portal URL if HTML parsing yields nothing.
    """
    form_data = urllib.parse.urlencode({
        "query":            query,
        "Tribunal":         "TS",
        "Tipo_Resolucion":  "Sentencia",
        "rows":             str(max_results),
    }).encode("utf-8")

    logger.info("CENDOJ search: %s", query)
    raw = _http_get(CENDOJ_URL, method="POST", data=form_data)
    if raw is None:
        logger.warning("CENDOJ returned no data; returning fallback URL")
        return [{
            "titulo":   "Búsqueda CENDOJ",
            "tribunal": "Tribunal Supremo",
            "fecha":    "",
            "resumen":  (
                f"No se pudo conectar a CENDOJ. Visita manualmente: "
                f"{CENDOJ_URL}?query={urllib.parse.quote(query)}"
            ),
        }]

    html = raw.decode("utf-8", errors="replace")
    parser = _CendojParser()
    try:
        parser.feed(html)
    except Exception as exc:  # noqa: BLE001
        logger.warning("CENDOJ HTML parse error: %s", exc)

    if not parser.snippets:
        logger.info("CENDOJ: no verdictresult divs found — returning fallback URL")
        return [{
            "titulo":   "Búsqueda CENDOJ (sin parsear)",
            "tribunal": "Tribunal Supremo",
            "fecha":    "",
            "resumen":  (
                f"Visita el resultado directamente en: "
                f"{CENDOJ_URL}?query={urllib.parse.quote(query)}"
            ),
        }]

    results: list[dict[str, str]] = []
    for snippet in parser.snippets[:max_results]:
        # Try to extract a date pattern YYYY from the snippet
        fecha_match = re.search(r"\b(19|20)\d{2}\b", snippet)
        fecha = fecha_match.group(0) if fecha_match else ""
        # First sentence as title heuristic
        first_line = snippet.split(".")[0][:120].strip()
        results.append({
            "titulo":   first_line or "Sentencia TS",
            "tribunal": "Tribunal Supremo",
            "fecha":    fecha,
            "resumen":  snippet[:300],
        })

    logger.info("CENDOJ search returned %d result(s)", len(results))
    return results


# ─── Tool 3: search_web ───────────────────────────────────────────────────────

_LEGAL_KEYWORDS = frozenset({
    "sentencia", "resolución", "jurisprudencia", "artículo", "ley",
    "real decreto", "código", "tribunal", "juzgado", "auto", "recurso",
    "apelación", "casación", "amparo", "normativa", "reglamento", "boe",
    "decreto", "edicto", "providencia", "notificación", "contrato",
    "herencia", "testamento", "hipoteca", "concurso", "insolvencia",
})

_DDGS_HTML_URL = "https://html.duckduckgo.com/html/"


class _DDGHtmlParser(HTMLParser):
    """Scrape result links and snippets from DuckDuckGo HTML endpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._in_title = False
        self._in_snippet = False
        self._pending_url = ""
        self._pending_title = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_dict = dict(attrs)
        css = attr_dict.get("class") or ""
        if tag == "a" and "result__a" in css:
            self._pending_url = attr_dict.get("href") or ""
            self._in_title = True
        elif tag == "a" and "result__snippet" in css:
            self._in_snippet = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._in_title = False
            self._in_snippet = False

    def handle_data(self, data: str) -> None:
        stripped = data.strip()
        if not stripped:
            return
        if self._in_title:
            self._pending_title = stripped
        elif self._in_snippet and self._pending_url:
            self.results.append({
                "titulo":   self._pending_title,
                "url":      self._pending_url,
                "snippet":  stripped[:300],
            })
            self._pending_url = ""
            self._pending_title = ""


def _refine_query(query: str) -> str:
    """Add site restrictions for legal queries; always remove shopping noise."""
    lower = query.lower()
    is_legal = any(kw in lower for kw in _LEGAL_KEYWORDS)
    extra = ""
    if is_legal:
        extra = " site:boe.es OR site:.gob.es OR filetype:pdf"
    noise = " -site:amazon.com -site:ebay.com -inurl:comprar -inurl:precio"
    return query + extra + noise


def search_web(
    query: str,
    max_results: int = 5,
) -> list[dict[str, str]]:
    """Web search with legal query refinement.

    Tries the ``duckduckgo-search`` (ddgs) package first; falls back to
    scraping the DuckDuckGo HTML endpoint with urllib.

    Parameters
    ----------
    query:       Search query (Spanish or mixed).
    max_results: Maximum results to return (default 5).

    Returns a list of dicts: {titulo, url, snippet}.
    """
    refined = _refine_query(query)
    logger.info("Web search (refined): %s", refined)

    # ── Try ddgs package ──────────────────────────────────────────────────────
    try:
        from duckduckgo_search import DDGS  # type: ignore[import]
        with DDGS() as ddgs:
            hits = list(ddgs.text(refined, max_results=max_results))
        results = [
            {
                "titulo":   h.get("title", ""),
                "url":      h.get("href", ""),
                "snippet":  (h.get("body") or "")[:300],
            }
            for h in hits
        ]
        logger.info("ddgs search returned %d result(s)", len(results))
        return results
    except ImportError:
        logger.debug("duckduckgo-search not installed — using HTML fallback")
    except Exception as exc:  # noqa: BLE001
        logger.warning("ddgs error: %s — falling back to HTML scraping", exc)

    # ── HTML scraping fallback ────────────────────────────────────────────────
    params = urllib.parse.urlencode({"q": refined})
    url = f"{_DDGS_HTML_URL}?{params}"
    raw = _http_get(url, headers={"Accept": "text/html"})
    if raw is None:
        logger.warning("DuckDuckGo HTML endpoint returned no data")
        return []

    html = raw.decode("utf-8", errors="replace")
    parser = _DDGHtmlParser()
    try:
        parser.feed(html)
    except Exception as exc:  # noqa: BLE001
        logger.warning("DDG HTML parse error: %s", exc)

    results = parser.results[:max_results]
    logger.info("DDG HTML scrape returned %d result(s)", len(results))
    return results


# ─── Tool 4: get_article ─────────────────────────────────────────────────────

LEY_DOCS: dict[str, str] = {
    "codigo_penal":           "BOE-A-1995-25444",
    "constitucion":           "BOE-A-1978-31229",
    "lec":                    "BOE-A-2000-323",
    "lecrim":                 "BOE-A-1882-6036",
    "estatuto_trabajadores":  "BOE-A-2015-11430",
    "lopj":                   "BOE-A-1985-12666",
}

BOE_ACT_URL = (
    "https://www.boe.es/buscar/act.php?id={doc_id}&tn=1&p=&acc=Elegir#a{articulo}"
)


def get_article(ley: str, articulo: str | int) -> str:
    """Fetch the text of *articulo* from the consolidated version of *ley*.

    Parameters
    ----------
    ley:      Normalised law name.  Must be one of the keys in LEY_DOCS or a
              fallback search_boe call is performed.
    articulo: Article number (int or string, e.g. 248 or "248").

    Returns the article text (up to 800 chars) or an error/fallback message.
    """
    articulo = str(articulo)
    ley_key = ley.lower().replace(" ", "_").replace("-", "_")
    logger.info("get_article: ley=%s articulo=%s", ley_key, articulo)

    if ley_key not in LEY_DOCS:
        logger.info("Ley '%s' not in LEY_DOCS — falling back to search_boe", ley_key)
        hits = search_boe(f"artículo {articulo} {ley.replace('_', ' ')}", max_results=1)
        if hits:
            return (
                f"[Artículo {articulo} de {ley} — resultado aproximado vía BOE]\n"
                f"{hits[0]['resumen']}"
            )
        return f"No se encontró información sobre el artículo {articulo} de {ley}."

    doc_id = LEY_DOCS[ley_key]
    url = BOE_ACT_URL.format(doc_id=doc_id, articulo=articulo)
    logger.info("Fetching BOE consolidated text: %s", url)

    raw = _http_get(url)
    if raw is None:
        return f"No se pudo obtener el artículo {articulo} de {ley} (error de red)."

    html = raw.decode("utf-8", errors="replace")

    # Try to isolate the fragment for this specific article.
    # BOE consolidated HTML uses <a name="a{N}"> anchors.
    anchor_pattern = re.compile(
        rf'<a[^>]*name=["\']?a{re.escape(articulo)}["\']?[^>]*>',
        re.IGNORECASE,
    )
    m = anchor_pattern.search(html)
    if m:
        fragment = html[m.start():m.start() + 4000]
    else:
        fragment = html

    text = _strip_html(fragment)
    # Trim to 800 chars
    if len(text) > 800:
        text = text[:800].rsplit(" ", 1)[0] + " [...]"

    if not text.strip():
        return f"Artículo {articulo} de {ley} no encontrado en el documento BOE."

    return text


# ─── Tool 5: calculate_plazo ─────────────────────────────────────────────────

# Fixed Spanish national public holidays (MM-DD format, year-independent).
# 12 official fiestas nacionales per RD 1110/2015 and successive annual BOEs.
_NATIONAL_HOLIDAYS_MD: frozenset[str] = frozenset({
    "01-01",  # Año Nuevo
    "01-06",  # Reyes Magos
    "04-18",  # Viernes Santo (approximate — variable; see note below)
    "05-01",  # Día del Trabajo
    "08-15",  # Asunción de la Virgen
    "10-12",  # Fiesta Nacional de España
    "11-01",  # Todos los Santos
    "12-06",  # Día de la Constitución
    "12-08",  # Inmaculada Concepción
    "12-25",  # Navidad
})

# Viernes Santo (Good Friday) is moveable.  Pre-computed 2020-2040.
_GOOD_FRIDAY: dict[int, str] = {
    2020: "04-10", 2021: "04-02", 2022: "04-15", 2023: "04-07",
    2024: "03-29", 2025: "04-18", 2026: "04-03", 2027: "03-26",
    2028: "04-14", 2029: "03-30", 2030: "04-19", 2031: "04-11",
    2032: "03-26", 2033: "04-15", 2034: "04-07", 2035: "03-23",
    2036: "04-11", 2037: "04-03", 2038: "04-23", 2039: "04-08",
    2040: "03-30",
}

_ES_MONTH_NAMES = [
    "", "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
]
_ES_WEEKDAY_NAMES = [
    "lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo",
]


def _is_holiday(d: date) -> bool:
    md = d.strftime("%m-%d")
    if md in _NATIONAL_HOLIDAYS_MD:
        return True
    # Viernes Santo
    gf = _GOOD_FRIDAY.get(d.year)
    return gf is not None and md == gf


def _format_date_es(d: date) -> str:
    """Return a Spanish human-readable date, e.g. 'lunes 14 de abril de 2025'."""
    wd = _ES_WEEKDAY_NAMES[d.weekday()]
    mo = _ES_MONTH_NAMES[d.month]
    return f"{wd} {d.day} de {mo} de {d.year}"


def calculate_plazo(
    fecha_inicio: str,
    dias: int,
    tipo: str = "habiles",
) -> dict[str, str]:
    """Compute a procedural deadline from *fecha_inicio*.

    Parameters
    ----------
    fecha_inicio: Start date in ISO format "YYYY-MM-DD".  The deadline begins
                  the *following* day (dies a quo non computatur).
    dias:         Number of days to count.
    tipo:         "habiles"   — working days (skips weekends + national holidays)
                  "naturales" — calendar days (counts all days including weekends)

    Returns a dict: {fecha_fin, dias_totales, tipo, detalle}.
    """
    try:
        start = date.fromisoformat(fecha_inicio)
    except ValueError as exc:
        return {"status": "error", "message": f"Fecha inválida: {exc}"}

    tipo_norm = tipo.lower().strip()
    if tipo_norm not in ("habiles", "naturales", "hábiles"):
        return {"status": "error", "message": f"tipo debe ser 'habiles' o 'naturales', no '{tipo}'"}
    use_habiles = tipo_norm in ("habiles", "hábiles")

    current = start
    counted = 0
    total_calendar = 0

    while counted < dias:
        current += timedelta(days=1)
        total_calendar += 1
        if use_habiles:
            # Skip weekends (5=Sat, 6=Sun) and national holidays
            if current.weekday() >= 5 or _is_holiday(current):
                continue
        counted += 1

    tipo_label = "hábiles" if use_habiles else "naturales"
    return {
        "fecha_fin":     current.isoformat(),
        "dias_totales":  str(total_calendar),
        "tipo":          tipo_label,
        "detalle":       _format_date_es(current),
    }


# ─── ToolRegistry ─────────────────────────────────────────────────────────────

class ToolRegistry:
    """Registry of callable tools with MCP-schema export and byte-stream I/O.

    The registry understands the 0xFF (ÿ) delimiter protocol:
      - parse_tool_call(bytes) → dict | None  — decode model tool request
      - format_result(dict)    → bytes         — encode result for injection

    Usage
    -----
        registry = ToolRegistry()
        registry.register(search_boe, "search_boe", "Busca en el BOE", {...})
        result = registry.execute("search_boe", query="contrato")
        schema = registry.to_mcp_schema()
    """

    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register(
        self,
        tool_fn: Any,
        name: str,
        description: str,
        params_schema: dict[str, Any],
    ) -> None:
        """Register a callable tool.

        Parameters
        ----------
        tool_fn:       Callable that implements the tool.
        name:          Tool name (used in TOOL/RESULT byte frames).
        description:   Human-readable description for MCP schema.
        params_schema: JSON-Schema ``properties`` dict for the parameters.
        """
        self._tools[name] = {
            "fn":          tool_fn,
            "description": description,
            "schema":      params_schema,
        }
        logger.debug("Registered tool '%s'", name)

    # ── Execution ─────────────────────────────────────────────────────────────

    def execute(self, name: str, **kwargs: Any) -> dict[str, Any]:
        """Execute a registered tool by name.

        Returns ``{"status": "ok", "data": ...}`` on success or
        ``{"status": "error", "message": ...}`` on failure.
        """
        if name not in self._tools:
            return {"status": "error", "message": f"Tool '{name}' not registered"}
        try:
            result = self._tools[name]["fn"](**kwargs)
            return {"status": "ok", "data": result}
        except TypeError as exc:
            return {"status": "error", "message": f"Invalid arguments for '{name}': {exc}"}
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unhandled error in tool '%s'", name)
            return {"status": "error", "message": str(exc)}

    # ── MCP schema ────────────────────────────────────────────────────────────

    def to_mcp_schema(self) -> list[dict[str, Any]]:
        """Return a list of MCP-compatible tool schema dicts."""
        schemas: list[dict[str, Any]] = []
        for name, meta in self._tools.items():
            schemas.append({
                "name":        name,
                "description": meta["description"],
                "inputSchema": {
                    "type":       "object",
                    "properties": meta["schema"],
                },
            })
        return schemas

    # ── Byte-stream I/O ───────────────────────────────────────────────────────

    @staticmethod
    def parse_tool_call(byte_sequence: bytes) -> dict[str, Any] | None:
        """Parse a ÿTOOL:{...}ÿ frame from a byte sequence.

        Returns the parsed JSON dict (must contain at least "name") or None if
        the input does not match the expected format.
        """
        # Find ÿTOOL: ... ÿ  (0xFF bytes as delimiters)
        pattern = re.compile(rb"\xff" + rb"TOOL:" + rb"(\{.*?\})" + rb"\xff", re.DOTALL)
        m = pattern.search(byte_sequence)
        if not m:
            return None
        try:
            payload = json.loads(m.group(1).decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            logger.warning("parse_tool_call: JSON decode error: %s", exc)
            return None
        if "name" not in payload:
            logger.warning("parse_tool_call: missing 'name' key in %s", payload)
            return None
        return payload

    @staticmethod
    def format_result(result: dict[str, Any]) -> bytes:
        """Encode *result* as a ÿRESULT:{...}ÿ byte frame.

        The JSON payload is encoded as UTF-8; the surrounding delimiter bytes
        are 0xFF, which is invalid UTF-8 and therefore unambiguous.
        """
        json_bytes = json.dumps(result, ensure_ascii=False).encode("utf-8")
        return DELIM + b"RESULT:" + json_bytes + DELIM


# ─── Build default registry ───────────────────────────────────────────────────

def build_registry() -> ToolRegistry:
    """Instantiate and return a ToolRegistry pre-loaded with all five tools."""
    registry = ToolRegistry()

    registry.register(
        search_boe,
        name="search_boe",
        description=(
            "Busca documentos en el BOE (Boletín Oficial del Estado) "
            "usando la API de datos abiertos de boe.es."
        ),
        params_schema={
            "query":       {"type": "string",  "description": "Consulta de búsqueda"},
            "year_from":   {"type": "integer", "description": "Año mínimo de publicación (opcional)"},
            "max_results": {"type": "integer", "description": "Número máximo de resultados (default 5)"},
        },
    )

    registry.register(
        search_cendoj,
        name="search_cendoj",
        description=(
            "Busca sentencias del Tribunal Supremo en el portal CENDOJ "
            "(Centro de Documentación Judicial del CGPJ)."
        ),
        params_schema={
            "query":       {"type": "string",  "description": "Consulta jurisprudencial"},
            "max_results": {"type": "integer", "description": "Número máximo de resultados (default 5)"},
        },
    )

    registry.register(
        search_web,
        name="search_web",
        description=(
            "Búsqueda web general con refinamiento automático para consultas legales. "
            "Usa DuckDuckGo."
        ),
        params_schema={
            "query":       {"type": "string",  "description": "Consulta de búsqueda"},
            "max_results": {"type": "integer", "description": "Número máximo de resultados (default 5)"},
        },
    )

    registry.register(
        get_article,
        name="get_article",
        description=(
            "Obtiene el texto de un artículo concreto de una ley española "
            "desde la versión consolidada del BOE."
        ),
        params_schema={
            "ley":      {"type": "string", "description": "Nombre normalizado de la ley (e.g. 'codigo_penal')"},
            "articulo": {"type": "string", "description": "Número de artículo (e.g. '248')"},
        },
    )

    registry.register(
        calculate_plazo,
        name="calculate_plazo",
        description=(
            "Calcula un plazo procesal en días hábiles o naturales desde una "
            "fecha de inicio, teniendo en cuenta festivos nacionales españoles."
        ),
        params_schema={
            "fecha_inicio": {"type": "string",  "description": "Fecha de inicio en formato YYYY-MM-DD"},
            "dias":         {"type": "integer", "description": "Número de días del plazo"},
            "tipo":         {"type": "string",  "description": "'habiles' o 'naturales' (default 'habiles')"},
        },
    )

    return registry


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Capibara Legal — tool executor CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python scripts/tool_executor.py --tool search_boe --query "recurso de amparo"
  python scripts/tool_executor.py --tool get_article --ley codigo_penal --articulo 248
  python scripts/tool_executor.py --tool calculate_plazo --fecha 2025-03-01 --dias 20 --tipo habiles
  python scripts/tool_executor.py --tool search_web --query "sentencia tribunal supremo daños morales 2024"
  python scripts/tool_executor.py --tool search_cendoj --query "responsabilidad civil médica"
""",
    )
    p.add_argument("--tool", required=True, help="Tool name to execute")
    p.add_argument("--query",    default=None, help="Search query (search_boe, search_cendoj, search_web)")
    p.add_argument("--ley",      default=None, help="Ley name (get_article)")
    p.add_argument("--articulo", default=None, help="Article number (get_article)")
    p.add_argument("--fecha",    default=None, help="Start date YYYY-MM-DD (calculate_plazo)")
    p.add_argument("--dias",     default=None, type=int, help="Number of days (calculate_plazo)")
    p.add_argument("--tipo",     default="habiles", help="'habiles' or 'naturales' (calculate_plazo)")
    p.add_argument("--year-from", dest="year_from", type=int, default=None, help="Year filter (search_boe)")
    p.add_argument("--max-results", dest="max_results", type=int, default=5, help="Max results")
    p.add_argument("--json", action="store_true", help="Output raw JSON instead of pretty-printed")
    p.add_argument("--mcp-schema", action="store_true", help="Print MCP tool schemas and exit")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    registry = build_registry()

    if args.mcp_schema:
        print(json.dumps(registry.to_mcp_schema(), ensure_ascii=False, indent=2))
        return

    tool = args.tool
    kwargs: dict[str, Any] = {}

    if tool == "search_boe":
        if not args.query:
            print("ERROR: --query required for search_boe", flush=True)
            raise SystemExit(1)
        kwargs["query"] = args.query
        if args.year_from:
            kwargs["year_from"] = args.year_from
        kwargs["max_results"] = args.max_results

    elif tool == "search_cendoj":
        if not args.query:
            print("ERROR: --query required for search_cendoj", flush=True)
            raise SystemExit(1)
        kwargs["query"] = args.query
        kwargs["max_results"] = args.max_results

    elif tool == "search_web":
        if not args.query:
            print("ERROR: --query required for search_web", flush=True)
            raise SystemExit(1)
        kwargs["query"] = args.query
        kwargs["max_results"] = args.max_results

    elif tool == "get_article":
        if not args.ley or not args.articulo:
            print("ERROR: --ley and --articulo required for get_article", flush=True)
            raise SystemExit(1)
        kwargs["ley"] = args.ley
        kwargs["articulo"] = args.articulo

    elif tool == "calculate_plazo":
        if not args.fecha or args.dias is None:
            print("ERROR: --fecha and --dias required for calculate_plazo", flush=True)
            raise SystemExit(1)
        kwargs["fecha_inicio"] = args.fecha
        kwargs["dias"] = args.dias
        kwargs["tipo"] = args.tipo

    else:
        print(f"ERROR: Unknown tool '{tool}'. Available: search_boe, search_cendoj, search_web, get_article, calculate_plazo")
        raise SystemExit(1)

    result = registry.execute(tool, **kwargs)

    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
