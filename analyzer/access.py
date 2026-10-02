import hmac
import ipaddress
import json
import os
import threading
from pathlib import Path


class AccessError(Exception):
    pass


class AccessSettings:
    """Who may use the container without a token, and the token for everyone
    else: the environment's, then whatever the panel saved over them."""

    def __init__(self, path, token="", network=None, host_ip=""):
        self._path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        self._token, self._network, self._host_ip = token, network, host_ip
        if self._path is None:
            return
        try:
            saved = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved = {}
        if isinstance(saved, dict):
            try:
                self._host_ip = str(self._validate_host(saved.get("host_ip")))
            except AccessError:
                pass
            try:
                self._network = self._validate_network(saved.get("network"), self._host_ip)
            except AccessError:
                pass
            try:
                self._token = self._validate_token(saved.get("token"))
            except AccessError:
                pass

    @staticmethod
    def _validate_host(value):
        if not isinstance(value, str):
            raise AccessError
        try:
            address = ipaddress.IPv4Address(value)
        except ValueError:
            raise AccessError from None
        if not address.is_private or address.is_loopback:
            raise AccessError
        return address

    @staticmethod
    def _validate_network(value, host_ip):
        if not isinstance(value, str):
            raise AccessError
        try:
            network = ipaddress.ip_network(value, strict=False)
            address = ipaddress.IPv4Address(host_ip)
        except ValueError:
            raise AccessError from None
        if (
            not isinstance(network, ipaddress.IPv4Network) or not network.is_private
            or not 16 <= network.prefixlen <= 30 or address not in network
        ):
            raise AccessError
        return network

    @staticmethod
    def _validate_token(value):
        if not isinstance(value, str):
            raise AccessError
        if value == "":
            return value
        value = value.strip()
        if not 16 <= len(value) <= 200 or any(c.isspace() or not c.isprintable() for c in value):
            raise AccessError
        return value

    def status(self):
        with self._lock:
            return {
                "configured": self._network is not None,
                "tokenSet": bool(self._token),
                "network": str(self._network) if self._network is not None else "",
                "hostIP": self._host_ip,
            }

    def update(self, host_ip, network, token=None):
        host_ip = str(self._validate_host(host_ip))
        network = self._validate_network(network, host_ip)
        if token is not None:
            token = self._validate_token(token)
        with self._lock:
            token = token or self._token
            if self._path is not None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self._path.with_suffix(".tmp")
                descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                    os.fchmod(file.fileno(), 0o600)
                    json.dump({"token": token, "network": str(network), "host_ip": host_ip}, file)
                os.replace(temporary, self._path)
            self._token, self._network, self._host_ip = token, network, host_ip

    def trusted(self, address, headers) -> bool:
        if any(header in headers for header in ("X-Forwarded-For", "X-Real-IP", "Forwarded")):
            return False
        try:
            address = ipaddress.IPv4Address(address)
        except (ValueError, TypeError):
            return False
        if str(address).rsplit(".", 1)[-1] == "1":
            return False
        with self._lock:
            network, host_ip = self._network, self._host_ip
        if network is not None:
            return address in network and str(address) != host_ip
        try:
            host = ipaddress.IPv4Address(headers.get("Host", "").split(":", 1)[0])
        except ValueError:
            return False
        return address.is_private and not address.is_loopback and host.is_private

    def token_matches(self, headers):
        with self._lock:
            token = self._token
        return bool(token) and hmac.compare_digest(
            headers.get("Authorization", "").encode("utf-8"),
            ("Bearer " + token).encode("utf-8"),
        )

    def authorized(self, address, headers) -> bool:
        return self.trusted(address, headers) or self.token_matches(headers)
