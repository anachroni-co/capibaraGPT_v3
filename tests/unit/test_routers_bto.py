"""Unit tests for core/routers/bto.py (BACKLOG-006).

bto.py is pure stdlib (no jax/flax), so we can exercise every branch.
The router package's ``__init__.py`` eagerly imports JAX-dependent modules,
so we load ``bto.py`` standalone via ``importlib`` to avoid pulling the
full ML stack into the test environment.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


import sys as _sys
_BTO_PATH = Path(__file__).resolve().parents[2] / "core" / "routers" / "bto.py"
_spec = importlib.util.spec_from_file_location("_bto_under_test", _BTO_PATH)
_bto = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
# Register before exec_module so @dataclass etc. can resolve the module.
_sys.modules["_bto_under_test"] = _bto
_spec.loader.exec_module(_bto)
BtoRouterV2 = _bto.BtoRouterV2


class _ConcreteRouter(BtoRouterV2):
    """Minimal concrete subclass so we can instantiate the ABC."""
    pass


class _Req:
    def __init__(self, path: str = "/default") -> None:
        self.path = path


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_construction():
    r = _ConcreteRouter()
    assert r.config == {}
    assert r.routes == {}
    assert r.initialized is False


def test_construction_with_config():
    cfg = {"mode": "fast", "timeout": 1.5}
    r = _ConcreteRouter(config=cfg)
    assert r.config is cfg
    assert r.routes == {}


# ---------------------------------------------------------------------------
# initialize()
# ---------------------------------------------------------------------------


def test_initialize_flips_flag_and_returns_true():
    r = _ConcreteRouter()
    assert r.initialize() is True
    assert r.initialized is True


# ---------------------------------------------------------------------------
# add_route / get_routes / remove_route
# ---------------------------------------------------------------------------


def test_add_route_and_get_routes_returns_copy():
    r = _ConcreteRouter()

    def handler(req):
        return {"ok": True}

    assert r.add_route("/foo", handler) is True
    assert "/foo" in r.routes

    got = r.get_routes()
    assert got == {"/foo": handler}
    # get_routes must return a copy - mutating the returned dict must not
    # affect the router's internal state.
    got["/bar"] = handler
    assert "/bar" not in r.routes


def test_remove_known_route_returns_true():
    r = _ConcreteRouter()
    r.add_route("/foo", lambda req: None)
    assert r.remove_route("/foo") is True
    assert "/foo" not in r.routes


def test_remove_unknown_route_returns_false():
    r = _ConcreteRouter()
    assert r.remove_route("/does-not-exist") is False


# ---------------------------------------------------------------------------
# route()
# ---------------------------------------------------------------------------


def test_route_dispatches_to_registered_handler():
    r = _ConcreteRouter()
    r.add_route("/ping", lambda req: {"pong": True})
    result = r.route(_Req("/ping"))
    assert result == {"pong": True}
    assert r.initialized is True, "route() must self-initialize"


def test_route_falls_back_to_default_handler_for_unknown_path():
    r = _ConcreteRouter()
    result = r.route(_Req("/unknown"))
    assert result["status"] == "success"
    assert "Default" in result["message"]


def test_route_uses_default_path_when_request_has_no_path_attr():
    r = _ConcreteRouter()

    class Bare:
        pass

    result = r.route(Bare())
    assert result["status"] == "success"


def test_route_returns_error_envelope_when_handler_raises():
    r = _ConcreteRouter()

    def boom(req):
        raise RuntimeError("handler exploded")

    r.add_route("/boom", boom)
    result = r.route(_Req("/boom"))
    assert result["status"] == "error"
    assert "handler exploded" in result["message"]


def test_add_route_is_idempotent_on_same_path():
    r = _ConcreteRouter()
    r.add_route("/x", lambda req: 1)
    r.add_route("/x", lambda req: 2)  # overwrites
    assert r.route(_Req("/x")) == 2
