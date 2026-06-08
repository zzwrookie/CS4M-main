from __future__ import annotations

import ipaddress
import re
from typing import Iterable


INTERNAL_ENV_CIDRS = (ipaddress.IPv4Network("128.55.12.0/24"),)
PRIVATE_CIDRS = (
    ipaddress.IPv4Network("10.0.0.0/8"),
    ipaddress.IPv4Network("192.168.0.0/16"),
    ipaddress.IPv4Network("172.16.0.0/12"),
)
ZERO_CIDR = ipaddress.IPv4Network("0.0.0.0/8")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_.:/-]+")


def _parse_ipv4(value: object) -> ipaddress.IPv4Address | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    parts = text.split(".")
    if len(parts) != 4:
        return None
    octets: list[int] = []
    for part in parts:
        if not part or not part.isdecimal():
            return None
        octet = int(part, 10)
        if octet < 0 or octet > 255:
            return None
        octets.append(octet)
    return ipaddress.IPv4Address(".".join(str(octet) for octet in octets))


def scope_ip(value: object) -> str:
    """Return the coarse scope token for an IPv4 address."""
    addr = _parse_ipv4(value)
    if addr is None:
        return "ip_unknown"
    if addr in ZERO_CIDR:
        return "ip_zero"
    if addr.is_loopback:
        return "ip_loopback"
    if any(addr in network for network in INTERNAL_ENV_CIDRS):
        return "ip_internal_env"
    if any(addr in network for network in PRIVATE_CIDRS):
        return "ip_private"
    return "ip_public"


def port_bucket(value: object) -> str:
    """Return the coarse port bucket token for a destination port."""
    if value is None:
        return "port_0"
    text = str(value).strip()
    if not text:
        return "port_0"
    try:
        port = int(text, 10)
    except ValueError:
        return "port_0"
    if port <= 0:
        return "port_0"
    if port <= 1023:
        return "port_system"
    if port <= 49151:
        return "port_registered"
    if port <= 65535:
        return "port_ephemeral"
    return "port_invalid"


def exact_ip_token(value: object) -> str:
    """Return an exact normalized IPv4 token, or ip_unknown for invalid input."""
    addr = _parse_ipv4(value)
    if addr is None:
        return "ip_unknown"
    return "ip_" + "_".join(str(addr).split("."))


def netflow_nll_role(dst_addr: object, dst_port: object) -> str:
    """Return the netflow role used by count-free NLL-style features."""
    return f"netflow|{scope_ip(dst_addr)}|{port_bucket(dst_port)}"


def netflow_residual_text(dst_addr: object) -> str:
    """Return residual netflow text with only the exact destination IP token."""
    return f"netflow {exact_ip_token(dst_addr)}"


def residual_text_tokens(value: object, max_tokens: int | None = None) -> tuple[str, ...]:
    """Return bounded natural residual text tokens for Word2Vec embedding."""
    if value is None:
        return ()
    limit = None if max_tokens is None else max(int(max_tokens), 0)
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(str(value).lower()):
        if limit is not None and len(tokens) >= limit:
            break
        tokens.append(match.group(0))
    return tuple(tokens)
