"""Who may talk to the server: a key outside loopback unless the operator
says otherwise, and a Host the server was told about - both decided at
start, both reported in /health.api.auth."""

from __future__ import annotations

import hmac
import ipaddress
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

# Names and addresses that mean this machine; a bind on one of them is
# reachable only from processes on it, so no key is asked for.
LOOPBACK_NAMES = ("localhost", "127.0.0.1", "::1")


class AccessError(ValueError):
    """The flags do not add up to a policy the server can start with."""


def is_loopback(host: str) -> bool:
    name = host.strip().strip("[]").lower()
    if name == "localhost":
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def _hostname(value: str) -> str:
    """The Host header without its port, lowercased: `name:8000`,
    `[::1]:8000`, `::1`."""
    value = value.strip().lower()
    if value.startswith("["):
        return value[1:].split("]", 1)[0]
    if value.count(":") == 1:
        return value.split(":", 1)[0]
    return value  # a bare IPv6 address, or no port


@dataclass(frozen=True)
class Access:
    # "loopback": the bind is local, nothing asked; "key": a bearer or
    # x-api-key must match; "skipped": open on purpose, by the flag.
    mode: str
    key: str | None
    hosts: tuple[str, ...]

    @classmethod
    def resolve(
        cls,
        host: str,
        api_key: str | None = None,
        skip_api_key: bool = False,
        allowed_hosts: Iterable[str] = (),
    ) -> Access:
        """The policy for a bind: a key or the explicit skip is required
        off loopback, refused with both options named."""
        if api_key and skip_api_key:
            raise AccessError("--api-key and --skip-api-key exclude each other")
        if api_key:
            mode = "key"
        elif skip_api_key:
            mode = "skipped"
        elif is_loopback(host):
            mode = "loopback"
        else:
            raise AccessError(
                f"--host {host} is reachable from other machines: pass --api-key "
                "<key> to require a key, or --skip-api-key to serve without one"
            )
        hosts = {_hostname(n) for n in LOOPBACK_NAMES}
        hosts.add(_hostname(host))
        hosts.update(_hostname(n) for n in allowed_hosts if n.strip())
        return cls(mode, api_key or None, tuple(sorted(hosts)))

    @classmethod
    def local(cls) -> Access:
        return cls.resolve("127.0.0.1")

    @property
    def wildcard(self) -> bool:
        """A bind on every interface: the Host clients send is not the
        bind address, so a name must be on the list."""
        return any(h in ("0.0.0.0", "::") for h in self.hosts)

    def host_allowed(self, header: str | None) -> bool:
        # No Host at all is HTTP/1.0; a browser always sends one, and the
        # rebinding attack needs the browser, so the bare request passes.
        if not header:
            return True
        return _hostname(header) in self.hosts

    def authorized(self, headers: Mapping[str, str]) -> bool:
        if self.mode != "key":
            return True
        offered = headers.get("x-api-key") or ""
        auth = headers.get("Authorization") or ""
        if not offered and auth[:7].lower() == "bearer ":
            offered = auth[7:].strip()
        # Compared as bytes: the header may carry any Latin-1 character
        # (http.server decodes it so), and compare_digest takes str only
        # in ASCII - a wrong key is a 401, not a dropped connection.
        return bool(offered) and hmac.compare_digest(
            offered.encode("utf-8"), (self.key or "").encode("utf-8")
        )

    def describe(self) -> dict:
        return {"mode": self.mode, "allowed_hosts": list(self.hosts)}
