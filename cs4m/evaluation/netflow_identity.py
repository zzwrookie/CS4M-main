"""Netflow canonical identity expansion for post-inference evaluation."""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from typing import Mapping


SERVICE_PORTS = {
    22: "ssh",
    53: "dns",
    80: "http",
    443: "https",
}


@dataclass(frozen=True)
class RemoteEndpoint:
    """Chosen remote endpoint and scope."""

    remote_ip: str
    remote_scope: str


@dataclass(frozen=True)
class NetflowIdentity:
    """Primary and explainability identities for a netflow node."""

    primary: str
    detail: str


def choose_non_local_endpoint(src_ip: str, dst_ip: str) -> RemoteEndpoint:
    """Choose the remote endpoint from two IP addresses."""
    src = _parse_ip(src_ip)
    dst = _parse_ip(dst_ip)
    if src is None and dst is None:
        return RemoteEndpoint("", "unknown")
    if src is None:
        assert dst is not None
        return RemoteEndpoint(str(dst), _scope(dst))
    if dst is None:
        return RemoteEndpoint(str(src), _scope(src))

    src_scope = _scope(src)
    dst_scope = _scope(dst)
    if src_scope == "private" and dst_scope == "public":
        return RemoteEndpoint(str(dst), dst_scope)
    if dst_scope == "private" and src_scope == "public":
        return RemoteEndpoint(str(src), src_scope)
    if src_scope in {"loopback", "link_local", "multicast"}:
        return RemoteEndpoint(str(src), src_scope)
    if dst_scope in {"loopback", "link_local", "multicast"}:
        return RemoteEndpoint(str(dst), dst_scope)
    return RemoteEndpoint(str(dst), dst_scope)


def canonical_netflow_identity(row: Mapping[str, object]) -> NetflowIdentity:
    """Build canonical netflow identity from endpoint fields."""
    endpoint = choose_non_local_endpoint(str(row.get("src_ip", "")), str(row.get("dst_ip", "")))
    service = _service_bucket(row)
    primary = f"netflow|remote_ip|{endpoint.remote_ip}"
    detail = f"netflow|{endpoint.remote_scope}|{service}|{endpoint.remote_ip}"
    return NetflowIdentity(primary=primary, detail=detail)


def expand_alerted_netflow_nodes(
    original_nodes: Mapping[str, Mapping[str, object]],
    alerted_canonical_identities: set[str],
) -> set[str]:
    """Return original node IDs covered by alerted canonical netflow identities."""
    covered: set[str] = set()
    for node_id, row in original_nodes.items():
        identity = canonical_netflow_identity(row)
        if identity.primary in alerted_canonical_identities:
            covered.add(str(node_id))
    return covered


def _parse_ip(value: str) -> ipaddress._BaseAddress | None:
    try:
        return ipaddress.ip_address(str(value).strip())
    except ValueError:
        return None


def _scope(value: ipaddress._BaseAddress) -> str:
    if value.is_multicast:
        return "multicast"
    if value.is_loopback:
        return "loopback"
    if value.is_link_local:
        return "link_local"
    if value.is_private:
        return "private"
    return "public"


def _service_bucket(row: Mapping[str, object]) -> str:
    for key in ("dst_port", "remote_port", "src_port"):
        try:
            port = int(str(row.get(key, "")).strip())
        except ValueError:
            continue
        if port in SERVICE_PORTS:
            return SERVICE_PORTS[port]
        if port >= 49152:
            return "ephemeral"
        return f"port_{port}"
    return "unknown_service"
