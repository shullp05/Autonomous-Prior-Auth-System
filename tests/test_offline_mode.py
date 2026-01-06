# test_offline_mode.py
from __future__ import annotations

import http.client
import os
import socket
import urllib.request
from typing import Any, Callable, Dict, Tuple

import pytest

from priorauth import offline_mode


def _snapshot_network_callables() -> Dict[str, Any]:
    """Capture originals so we can restore after tests (offline_mode patches globals)."""
    snap: Dict[str, Any] = {
        "socket_connect": socket.socket.connect,
        "socket_connect_ex": getattr(socket.socket, "connect_ex", None),
        "socket_sendto": getattr(socket.socket, "sendto", None),
        "socket_sendmsg": getattr(socket.socket, "sendmsg", None),
        "create_connection": socket.create_connection,
        "getaddrinfo": socket.getaddrinfo,
        "urlopen": urllib.request.urlopen,
        "http_connect": http.client.HTTPConnection.connect,
    }
    # requests is optional
    try:
        import requests.sessions  # type: ignore

        snap["requests_request"] = requests.sessions.Session.request
    except Exception:
        snap["requests_request"] = None
    return snap


def _skip_if_socket_denied(sock_type: int) -> None:
    try:
        s = socket.socket(socket.AF_INET, sock_type)
    except PermissionError:
        pytest.skip("Socket creation blocked in this environment; cannot validate loopback UDP.")
    else:
        s.close()


def _restore_network_callables(snap: Dict[str, Any]) -> None:
    socket.socket.connect = snap["socket_connect"]  # type: ignore[assignment]
    if snap.get("socket_connect_ex") is not None:
        socket.socket.connect_ex = snap["socket_connect_ex"]  # type: ignore[assignment]
    if snap.get("socket_sendto") is not None:
        socket.socket.sendto = snap["socket_sendto"]  # type: ignore[assignment]
    if snap.get("socket_sendmsg") is not None:
        socket.socket.sendmsg = snap["socket_sendmsg"]  # type: ignore[assignment]

    socket.create_connection = snap["create_connection"]  # type: ignore[assignment]
    socket.getaddrinfo = snap["getaddrinfo"]  # type: ignore[assignment]
    urllib.request.urlopen = snap["urlopen"]  # type: ignore[assignment]
    http.client.HTTPConnection.connect = snap["http_connect"]  # type: ignore[assignment]

    if snap.get("requests_request") is not None:
        try:
            import requests.sessions  # type: ignore

            requests.sessions.Session.request = snap["requests_request"]  # type: ignore[assignment]
        except Exception:
            pass


@pytest.fixture(autouse=True)
def _isolate_offline_patches(monkeypatch: pytest.MonkeyPatch):
    """
    offline_mode.enforce_offline() monkeypatches process-wide callables.
    We must restore after each test to prevent cross-test contamination.
    """
    snap = _snapshot_network_callables()
    yield
    _restore_network_callables(snap)
    # Clean env so other tests don't unintentionally enable offline mode
    monkeypatch.delenv("PA_OFFLINE_MODE", raising=False)
    monkeypatch.delenv("OFFLINE_MODE", raising=False)
    monkeypatch.delenv("PA_OFFLINE_ALLOW_LOCALHOST", raising=False)


def test_offline_disabled_does_not_patch(monkeypatch: pytest.MonkeyPatch):
    orig_connect = socket.socket.connect
    orig_create = socket.create_connection

    monkeypatch.setenv("PA_OFFLINE_MODE", "false")
    offline_mode.enforce_offline()

    assert socket.socket.connect is orig_connect
    assert socket.create_connection is orig_create


def test_offline_enabled_patches_core_tcp_dns(monkeypatch: pytest.MonkeyPatch):
    """
    This is an introspection test (no real network).
    It ensures offline mode actually patches the primitives that matter.
    """
    snap = _snapshot_network_callables()
    monkeypatch.setenv("PA_OFFLINE_MODE", "true")
    monkeypatch.setenv("PA_OFFLINE_ALLOW_LOCALHOST", "true")

    offline_mode.enforce_offline()

    assert socket.socket.connect is not snap["socket_connect"], "connect() must be patched"
    assert socket.create_connection is not snap["create_connection"], "create_connection() must be patched"

    # These MUST be patched to prevent bypasses and caller-breakage
    assert socket.getaddrinfo is not snap["getaddrinfo"], "getaddrinfo() must be patched"

    # Optional in some implementations, but strongly recommended:
    if snap.get("socket_connect_ex") is not None:
        assert socket.socket.connect_ex is not snap["socket_connect_ex"], "connect_ex() must be patched"


def test_offline_enabled_patches_udp_bypass(monkeypatch: pytest.MonkeyPatch):
    """
    No real egress. We enforce that sendto/sendmsg are monkeypatched
    so UDP cannot bypass offline mode.
    """
    snap = _snapshot_network_callables()
    monkeypatch.setenv("PA_OFFLINE_MODE", "true")
    monkeypatch.setenv("PA_OFFLINE_ALLOW_LOCALHOST", "true")

    offline_mode.enforce_offline()

    assert socket.socket.sendto is not snap["socket_sendto"], "sendto() must be patched to block UDP bypass"
    # sendmsg may not exist on some platforms; enforce if present
    if snap.get("socket_sendmsg") is not None:
        assert socket.socket.sendmsg is not snap["socket_sendmsg"], "sendmsg() must be patched to block UDP bypass"


def test_offline_blocks_external_dns_with_standard_error_types(monkeypatch: pytest.MonkeyPatch):
    """
    Must raise socket.gaierror/OSError types, not RuntimeError/OfflineModeError,
    so callers that expect OS-like failures don't break.
    """
    monkeypatch.setenv("PA_OFFLINE_MODE", "true")
    monkeypatch.setenv("PA_OFFLINE_ALLOW_LOCALHOST", "true")
    offline_mode.enforce_offline()

    with pytest.raises((socket.gaierror, OSError)):
        socket.getaddrinfo("example.com", 443)


def test_offline_allows_localhost_aliases(monkeypatch: pytest.MonkeyPatch):
    """
    Offline mode must NOT block localhost variants, otherwise it can brick local Ollama calls.
    We accept connection refused; we only reject "offline blocked" type failures.
    """
    monkeypatch.setenv("PA_OFFLINE_MODE", "true")
    monkeypatch.setenv("PA_OFFLINE_ALLOW_LOCALHOST", "true")
    offline_mode.enforce_offline()

    # These should be treated as loopback/localhost
    localhost_variants = ["localhost", "localhost.", "localhost.localdomain", "127.0.0.1", "::1"]

    for host in localhost_variants:
        try:
            s = socket.create_connection((host, 9), timeout=0.1)  # port 9 usually closed; refusal is fine
            s.close()
        except Exception as e:
            # It's okay if the port is closed; it's NOT okay if offline mode blocks it.
            msg = str(e).lower()
            assert "offline" not in msg, f"{host} should not be blocked by offline mode: {e!r}"


def test_offline_udp_allows_loopback_send(monkeypatch: pytest.MonkeyPatch):
    """
    Loopback UDP should remain allowed.
    This does not egress; it stays on loopback.
    """
    monkeypatch.setenv("PA_OFFLINE_MODE", "true")
    monkeypatch.setenv("PA_OFFLINE_ALLOW_LOCALHOST", "true")
    offline_mode.enforce_offline()

    _skip_if_socket_denied(socket.SOCK_DGRAM)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sent = s.sendto(b"ping", ("127.0.0.1", 9999))
        assert sent == 4
    finally:
        s.close()
