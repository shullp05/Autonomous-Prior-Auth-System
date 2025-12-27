"""
offline_mode.py - Hard Offline/No-Egress Guardrails (Localhost-Safe)

Purpose
-------
Opt-in "no outbound network" guard intended for:
- CI environments
- Portfolio/demo runs where you must prove **no external network egress**
- "dead-man switch" scenarios (block when enabled)

Activation
----------
Offline mode activates when:
- PA_OFFLINE_MODE=true  (or OFFLINE_MODE=true)

If you want to enforce offline mode programmatically regardless of env:
- enforce_offline(force=True)

Localhost Safety
----------------
When enabled, this guard **allows loopback** by default:
- localhost / localhost. / localhost.localdomain
- 127.0.0.1
- ::1
This preserves local Ollama calls.

Threat Model Notes
------------------
Blocking only connect() is NOT sufficient; UDP datagrams can be sent via
sendto()/sendmsg() without connecting. This implementation blocks both.

What gets patched (when enabled)
--------------------------------
- socket.socket.connect / connect_ex
- socket.socket.sendto / sendmsg (if available)
- socket.getaddrinfo
- socket.create_connection
- urllib.request.urlopen (best-effort)
- requests.Session.request (best-effort)

Env Configuration
-----------------
PA_OFFLINE_MODE / OFFLINE_MODE:
    truthy => enabled

PA_OFFLINE_ALLOW_LOCALHOST / OFFLINE_ALLOW_LOCALHOST:
    truthy (default) => allow loopback

PA_OFFLINE_ALLOWLIST / OFFLINE_ALLOWLIST:
    Comma-separated additional allowed hosts/IPs (exact match)
    Example: "my-internal-host,10.0.0.5"

PA_OFFLINE_STRICT_UNKNOWN_HOST / OFFLINE_STRICT_UNKNOWN_HOST:
    truthy (default) => missing/unknown host info is blocked

Author: Peter Shull, PharmD
License: MIT
"""

from __future__ import annotations

import errno
import ipaddress
import os
import socket
import urllib.parse
from dataclasses import dataclass
from typing import Any, Optional


class OfflineModeError(PermissionError):
    """Raised when offline mode blocks a network action."""


@dataclass(frozen=True)
class OfflineConfig:
    enabled: bool
    allow_localhost: bool
    allowlist: frozenset[str]
    strict_unknown_host: bool


def _env_truthy(val: str | None) -> bool:
    return (val or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def is_offline_enabled() -> bool:
    """True if PA_OFFLINE_MODE or OFFLINE_MODE is truthy."""
    return _env_truthy(os.getenv("PA_OFFLINE_MODE")) or _env_truthy(os.getenv("OFFLINE_MODE"))


def get_offline_config() -> OfflineConfig:
    allow_localhost = _env_truthy(
        os.getenv("PA_OFFLINE_ALLOW_LOCALHOST", os.getenv("OFFLINE_ALLOW_LOCALHOST", "true"))
    )
    allowlist_raw = os.getenv("PA_OFFLINE_ALLOWLIST", os.getenv("OFFLINE_ALLOWLIST", "")).strip()
    allowlist = frozenset(h.strip().lower().strip(".") for h in allowlist_raw.split(",") if h.strip())
    strict_unknown = _env_truthy(
        os.getenv("PA_OFFLINE_STRICT_UNKNOWN_HOST", os.getenv("OFFLINE_STRICT_UNKNOWN_HOST", "true"))
    )
    return OfflineConfig(
        enabled=is_offline_enabled(),
        allow_localhost=allow_localhost,
        allowlist=allowlist,
        strict_unknown_host=strict_unknown,
    )


_LOOPBACK_ALIASES = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
    }
)


def _normalize_host(host: str) -> str:
    h = (host or "").strip().lower()
    # strip brackets for IPv6 URLs like http://[::1]:11434
    if h.startswith("[") and h.endswith("]"):
        h = h[1:-1]
    # remove trailing dot (FQDN style)
    h = h.strip(".")
    return h


def is_loopback_host(host: str) -> bool:
    h = _normalize_host(host)
    if not h:
        return False
    if h in _LOOPBACK_ALIASES:
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def _is_allowed_host(host: str, cfg: OfflineConfig) -> bool:
    h = _normalize_host(host)
    if not h:
        return False
    if cfg.allow_localhost and is_loopback_host(h):
        return True
    if h in cfg.allowlist:
        return True
    return False


def _raise_blocked(op: str, host: str | None) -> None:
    msg = f"Offline mode blocked outbound network ({op}) to host={host!r}"
    raise OfflineModeError(errno.EACCES, msg)


def _block_if_needed(host: Optional[str], cfg: OfflineConfig, op: str) -> None:
    if not cfg.enabled:
        return
    if host is None:
        if cfg.strict_unknown_host:
            _raise_blocked(op, host)
        return
    if _is_allowed_host(host, cfg):
        return
    _raise_blocked(op, host)


def _extract_host_from_sockaddr(address: Any) -> Optional[str]:
    """
    Best-effort extraction of host from a socket address:
    - AF_INET/AF_INET6: (host, port, ...) tuples
    - AF_UNIX: str path => local => returns None
    """
    if address is None:
        return None
    if isinstance(address, (str, bytes)):
        # AF_UNIX path or something path-like. Treat as local.
        return None
    if isinstance(address, tuple) and len(address) >= 1:
        host = address[0]
        if isinstance(host, bytes):
            try:
                host = host.decode("utf-8", "ignore")
            except Exception:
                host = str(host)
        return str(host)
    return None


_PATCHED = False

_ORIG_SOCKET_CONNECT = None
_ORIG_SOCKET_CONNECT_EX = None
_ORIG_SOCKET_SENDTO = None
_ORIG_SOCKET_SENDMSG = None
_ORIG_GETADDRINFO = None
_ORIG_CREATE_CONNECTION = None
_ORIG_URLOPEN = None
_ORIG_REQUESTS_SESSION_REQUEST = None


def _patch_socket(cfg: OfflineConfig) -> None:
    global _ORIG_SOCKET_CONNECT, _ORIG_SOCKET_CONNECT_EX, _ORIG_SOCKET_SENDTO, _ORIG_SOCKET_SENDMSG
    global _ORIG_GETADDRINFO, _ORIG_CREATE_CONNECTION

    if _ORIG_SOCKET_CONNECT is None:
        _ORIG_SOCKET_CONNECT = socket.socket.connect
    if _ORIG_SOCKET_CONNECT_EX is None:
        _ORIG_SOCKET_CONNECT_EX = socket.socket.connect_ex
    if _ORIG_SOCKET_SENDTO is None:
        _ORIG_SOCKET_SENDTO = socket.socket.sendto
    if hasattr(socket.socket, "sendmsg") and _ORIG_SOCKET_SENDMSG is None:
        _ORIG_SOCKET_SENDMSG = getattr(socket.socket, "sendmsg")
    if _ORIG_GETADDRINFO is None:
        _ORIG_GETADDRINFO = socket.getaddrinfo
    if _ORIG_CREATE_CONNECTION is None:
        _ORIG_CREATE_CONNECTION = socket.create_connection

    def patched_connect(self: socket.socket, address: Any) -> Any:
        host = _extract_host_from_sockaddr(address)
        if host is not None:
            _block_if_needed(host, cfg, op="socket.connect")
        return _ORIG_SOCKET_CONNECT(self, address)  # type: ignore[misc]

    def patched_connect_ex(self: socket.socket, address: Any) -> int:
        host = _extract_host_from_sockaddr(address)
        if host is not None:
            try:
                _block_if_needed(host, cfg, op="socket.connect_ex")
            except OfflineModeError:
                return errno.EACCES
        return _ORIG_SOCKET_CONNECT_EX(self, address)  # type: ignore[misc]

    def patched_sendto(self: socket.socket, data: Any, address: Any, *args: Any) -> Any:
        host = _extract_host_from_sockaddr(address)
        if host is not None:
            _block_if_needed(host, cfg, op="socket.sendto")
        return _ORIG_SOCKET_SENDTO(self, data, address, *args)  # type: ignore[misc]

    socket.socket.connect = patched_connect  # type: ignore[assignment]
    socket.socket.connect_ex = patched_connect_ex  # type: ignore[assignment]
    socket.socket.sendto = patched_sendto  # type: ignore[assignment]

    if _ORIG_SOCKET_SENDMSG is not None:

        def patched_sendmsg(self: socket.socket, *args: Any, **kwargs: Any) -> Any:
            # sendmsg(buffers[, ancdata[, flags[, address]]])
            address = None
            if "address" in kwargs:
                address = kwargs.get("address")
            elif len(args) >= 4:
                address = args[3]
            host = _extract_host_from_sockaddr(address)
            if host is not None:
                _block_if_needed(host, cfg, op="socket.sendmsg")
            return _ORIG_SOCKET_SENDMSG(self, *args, **kwargs)  # type: ignore[misc]

        setattr(socket.socket, "sendmsg", patched_sendmsg)  # type: ignore[arg-type]

    def patched_getaddrinfo(host: Any, port: Any, *args: Any, **kwargs: Any) -> Any:
        # host can be None for passive binds; allow those.
        if host is None:
            return _ORIG_GETADDRINFO(host, port, *args, **kwargs)  # type: ignore[misc]

        host_str = str(host)
        if cfg.enabled and not _is_allowed_host(host_str, cfg):
            # Mimic typical DNS resolution failure
            raise socket.gaierror(socket.EAI_NONAME, "Name or service not known (offline mode)")
        return _ORIG_GETADDRINFO(host, port, *args, **kwargs)  # type: ignore[misc]

    def patched_create_connection(address: Any, *args: Any, **kwargs: Any) -> socket.socket:
        host = _extract_host_from_sockaddr(address)
        if host is not None:
            _block_if_needed(host, cfg, op="socket.create_connection")
        return _ORIG_CREATE_CONNECTION(address, *args, **kwargs)  # type: ignore[misc]

    socket.getaddrinfo = patched_getaddrinfo  # type: ignore[assignment]
    socket.create_connection = patched_create_connection  # type: ignore[assignment]


def _patch_urllib(cfg: OfflineConfig) -> None:
    global _ORIG_URLOPEN
    try:
        import urllib.request as _urllib_request
    except Exception:
        return

    if not hasattr(_urllib_request, "urlopen"):
        return

    if _ORIG_URLOPEN is None:
        _ORIG_URLOPEN = _urllib_request.urlopen

    def patched_urlopen(url: Any, *args: Any, **kwargs: Any) -> Any:
        full_url = getattr(url, "full_url", None) or str(url)
        parsed = urllib.parse.urlparse(full_url)
        _block_if_needed(parsed.hostname, cfg, op="urllib.request.urlopen")
        return _ORIG_URLOPEN(url, *args, **kwargs)  # type: ignore[misc]

    _urllib_request.urlopen = patched_urlopen  # type: ignore[assignment]


def _patch_requests(cfg: OfflineConfig) -> None:
    global _ORIG_REQUESTS_SESSION_REQUEST
    try:
        import requests  # type: ignore
    except Exception:
        return

    if _ORIG_REQUESTS_SESSION_REQUEST is None:
        _ORIG_REQUESTS_SESSION_REQUEST = requests.sessions.Session.request

    def patched_requests_request(self: Any, method: str, url: str, *args: Any, **kwargs: Any) -> Any:
        parsed = urllib.parse.urlparse(url)
        _block_if_needed(parsed.hostname, cfg, op="requests.Session.request")
        return _ORIG_REQUESTS_SESSION_REQUEST(self, method, url, *args, **kwargs)  # type: ignore[misc]

    requests.sessions.Session.request = patched_requests_request  # type: ignore[assignment]


def disable_offline() -> None:
    """Restore patched functions back to their originals (useful for tests/dev)."""
    global _PATCHED
    if not _PATCHED:
        return

    if _ORIG_SOCKET_CONNECT is not None:
        socket.socket.connect = _ORIG_SOCKET_CONNECT  # type: ignore[assignment]
    if _ORIG_SOCKET_CONNECT_EX is not None:
        socket.socket.connect_ex = _ORIG_SOCKET_CONNECT_EX  # type: ignore[assignment]
    if _ORIG_SOCKET_SENDTO is not None:
        socket.socket.sendto = _ORIG_SOCKET_SENDTO  # type: ignore[assignment]
    if _ORIG_SOCKET_SENDMSG is not None and hasattr(socket.socket, "sendmsg"):
        setattr(socket.socket, "sendmsg", _ORIG_SOCKET_SENDMSG)

    if _ORIG_GETADDRINFO is not None:
        socket.getaddrinfo = _ORIG_GETADDRINFO  # type: ignore[assignment]
    if _ORIG_CREATE_CONNECTION is not None:
        socket.create_connection = _ORIG_CREATE_CONNECTION  # type: ignore[assignment]

    try:
        import urllib.request as _urllib_request

        if _ORIG_URLOPEN is not None:
            _urllib_request.urlopen = _ORIG_URLOPEN  # type: ignore[assignment]
    except Exception:
        pass

    try:
        import requests  # type: ignore

        if _ORIG_REQUESTS_SESSION_REQUEST is not None:
            requests.sessions.Session.request = _ORIG_REQUESTS_SESSION_REQUEST  # type: ignore[assignment]
    except Exception:
        pass

    _PATCHED = False


def enforce_offline(*, force: bool = False) -> OfflineConfig:
    """
    Enable offline mode (patch network functions) if configured or forced.

    - Default: activates only if PA_OFFLINE_MODE/OFFLINE_MODE is truthy.
    - force=True: activates regardless of env vars.
    """
    global _PATCHED
    cfg = get_offline_config()
    if force:
        cfg = OfflineConfig(
            enabled=True,
            allow_localhost=cfg.allow_localhost,
            allowlist=cfg.allowlist,
            strict_unknown_host=cfg.strict_unknown_host,
        )

    if not cfg.enabled:
        return cfg
    if _PATCHED:
        return cfg

    _patch_socket(cfg)
    _patch_urllib(cfg)
    _patch_requests(cfg)
    _PATCHED = True
    return cfg
