"""Isolated tests for core.routing module."""

import pytest
from core.routing import CoreHttpRouter, Router, create_router


class TestCoreHttpRouter:
    """Tests for CoreHttpRouter class."""

    def test_init_empty_routes(self):
        router = CoreHttpRouter()
        assert router.routes == {}
        assert router.default_route is None

    def test_add_route(self):
        router = CoreHttpRouter()
        handler = lambda: {"ok": True}
        router.add_route("/test", handler)
        assert "/test" in router.routes
        assert router.routes["/test"] is handler

    def test_route_to_registered_path(self):
        router = CoreHttpRouter()
        router.add_route("/hello", lambda: {"msg": "hi"})
        result = router.route("/hello")
        assert result == {"msg": "hi"}

    def test_route_with_kwargs(self):
        router = CoreHttpRouter()
        router.add_route("/echo", lambda text=None: {"echo": text})
        result = router.route("/echo", text="world")
        assert result == {"echo": "world"}

    def test_route_no_match_no_default(self):
        router = CoreHttpRouter()
        result = router.route("/missing")
        assert result["status"] == "no_route"
        assert result["path"] == "/missing"

    def test_default_route_fallback(self):
        router = CoreHttpRouter()
        router.set_default_route(lambda **kw: {"default": True, **kw})
        result = router.route("/unknown", x=1)
        assert result["default"] is True
        assert result["x"] == 1

    def test_overwrite_route(self):
        router = CoreHttpRouter()
        router.add_route("/x", lambda: {"v": 1})
        router.add_route("/x", lambda: {"v": 2})
        assert router.route("/x") == {"v": 2}

    def test_multiple_routes(self):
        router = CoreHttpRouter()
        router.add_route("/a", lambda: "a")
        router.add_route("/b", lambda: "b")
        assert router.route("/a") == "a"
        assert router.route("/b") == "b"


class TestRouterAlias:
    def test_router_is_core_http_router(self):
        assert Router is CoreHttpRouter


class TestCreateRouter:
    def test_factory_returns_router(self):
        router = create_router()
        assert isinstance(router, CoreHttpRouter)

    def test_health_endpoint(self):
        router = create_router()
        result = router.route("/health")
        assert result == {"status": "healthy"}

    def test_info_endpoint(self):
        router = create_router()
        info = router.route("/info")
        assert info["name"] == "CapibaraGPT"
        assert info["version"] == "3.0"
        assert "mesh" in info
        assert "dtype" in info

    def test_custom_route_on_factory_router(self):
        router = create_router()
        router.add_route("/custom", lambda: {"custom": True})
        assert router.route("/custom") == {"custom": True}
        # default routes still work
        assert router.route("/health")["status"] == "healthy"
