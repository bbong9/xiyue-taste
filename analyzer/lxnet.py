"""Network requests for source scripts, by the rules of iOS LXNetworkPolicy and LXNetworkClient.

The Node runner never touches the network: a script's lx.request reaches this
module through the runner protocol. This is the trust boundary: addresses,
sizes, headers and timeouts are all checked here, and Node does not repeat
them. Every hop is checked before it is connected, and the connection goes
only to the addresses that passed (no second lookup to rebind). No cookies are
kept. Nothing is logged here; callers log only the host name and status code.
"""

import collections
import http.client
import ipaddress
import json
import math
import posixpath
import re
import socket
import ssl
import threading
import urllib.parse
import zlib

MAX_URL_BYTES = 16 * 1024
MAX_OPTIONS_BYTES = 64 * 1024
MAX_REQUEST_BYTES = 256 * 1024
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_ADDRESSES = 64
MAX_REDIRECTS = 6
MAX_HEADERS = 32
MAX_HEADER_BYTES = 16 * 1024
MAX_RESPONSE_HEADERS = 32
MAX_RESPONSE_HEADER_CHARS = 2048
METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD")
FORBIDDEN_HEADERS = frozenset(("authorization", "cookie", "proxy-authorization", "host"))
HIDDEN_RESPONSE_HEADERS = frozenset(("set-cookie", "authorization"))
REDIRECT_STATUSES = (301, 302, 303, 307, 308)
DEFAULT_ACCEPT = "application/json"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36"
)
# iOS isUnplayableMediaFormat, the whole list: encrypted platform formats and Ogg.
# A source answering a tier with one of these has not delivered that tier.
UNPLAYABLE_EXTENSIONS = frozenset((
    "mgg", "mgg0", "mgg1", "mggl", "mflac", "mflac0", "mflac1", "mmp4",
    "qmc0", "qmc2", "qmc3", "qmcflac", "qmcogg", "tkm", "bkcmp3", "bkcflac",
    "kgm", "kgma", "vpr", "ncm", "ogg", "oga", "opus",
))
_CHUNK = 64 * 1024
# The ranges iOS blockedIPv4Class / blockedIPv6Class refuse; Python must also call the address global.
_BLOCKED_NETWORKS = tuple(ipaddress.ip_network(value) for value in (
    "0.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "127.0.0.0/8",
    "169.254.0.0/16", "100.64.0.0/10", "192.0.0.0/16", "198.18.0.0/15", "224.0.0.0/3",
    "::/128", "::1/128", "fc00::/7", "fe80::/10", "fec0::/10", "ff00::/8",
))
_TOKEN = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_LONE_PERCENT = re.compile(r"%(?![0-9A-Fa-f]{2})")
_TLS = ssl.create_default_context()

Reply = collections.namedtuple("Reply", "error response_json body_text host status")
_Target = collections.namedtuple("_Target", "scheme host connect_host port path addresses")


class NetError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


class Abort:
    """Lets another thread (a cancel, the overall deadline) cut a request short.

    It shuts down the socket in use, so a blocked read returns at once.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._sock = None
        self.reason = None

    def attach(self, sock):
        with self._lock:
            self._sock = sock
            aborted = self.reason is not None
        if aborted:
            _shutdown(sock)

    def abort(self, reason):
        with self._lock:
            if self.reason is None:
                self.reason = reason
            sock = self._sock
        if sock is not None:
            _shutdown(sock)

    def check(self):
        if self.reason is not None:
            raise NetError(self.reason)


def _shutdown(sock):
    try:
        # The plain socket call, so a TLS socket is cut without touching its TLS state.
        socket.socket.shutdown(sock, socket.SHUT_RDWR)
    except OSError:
        pass


def is_public(address):
    """Outside every range iOS refuses, and global to Python too.

    An IPv4-mapped or IPv4-compatible IPv6 address counts as the IPv4 inside it.
    """
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    if any(ip in network for network in _BLOCKED_NETWORKS):
        return False
    if ip.version == 6:
        packed = ip.packed
        if packed[:10] == bytes(10) and packed[10:12] in (b"\xff\xff", b"\x00\x00"):
            return is_public(str(ipaddress.IPv4Address(packed[12:])))
    return ip.is_global


def _utf8_length(text):
    return len(text.encode("utf-8", "surrogatepass"))


def _target(url, *, allow_http, allow_private):
    """iOS validatedHost and systemStackPreflight: the shape, the host name, every address it resolves to."""
    if not isinstance(url, str) or _utf8_length(url) > MAX_URL_BYTES:
        raise NetError("request_rejected")
    if url != url.strip() or _CONTROL.search(url) or "#" in url:
        raise NetError("request_rejected")
    try:
        parts = urllib.parse.urlsplit(url)
        port = parts.port
    except ValueError:
        raise NetError("request_rejected") from None
    scheme = parts.scheme.lower()
    host = parts.hostname or ""
    if scheme != "https" and not (allow_http and scheme == "http"):
        raise NetError("request_rejected")
    if not host or "@" in parts.netloc:
        raise NetError("request_rejected")
    if not allow_private and (host in ("localhost", "::1") or host.endswith((".local", ".internal"))):
        raise NetError("request_rejected")
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is None:
        try:
            name = host.encode("idna").decode("ascii")
        except UnicodeError:
            raise NetError("request_rejected") from None
        connect_host = name
    else:
        name = host
        connect_host = f"[{host}]" if literal.version == 6 else host
    port = port or (443 if scheme == "https" else 80)
    flags = socket.AI_NUMERICHOST if literal is not None else socket.AI_ADDRCONFIG
    try:
        found = socket.getaddrinfo(name, port, type=socket.SOCK_STREAM, flags=flags)
    except (OSError, UnicodeError, ValueError):
        raise NetError("request_rejected") from None
    addresses = {}
    for family, _, _, _, sockaddr in found:
        addresses.setdefault(sockaddr[0], (family, sockaddr))
    if not addresses or len(addresses) > MAX_ADDRESSES:
        raise NetError("request_rejected")
    if not allow_private and not all(is_public(address) for address in addresses):
        raise NetError("request_rejected")
    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    # Like Foundation, characters that may not stand in a URL are percent-encoded; valid escapes are kept.
    try:
        path = urllib.parse.quote(_LONE_PERCENT.sub("%25", path), safe="!$%&'()*+,/:;=?@~")
    except UnicodeError:
        raise NetError("request_rejected") from None
    return _Target(scheme, name, connect_host, port, path, list(addresses.values()))


def _pinned(addresses, abort):
    """http.client's connect, but only to the addresses already checked."""

    def create(_address, timeout=None, _source_address=None):
        failure = None
        for family, sockaddr in addresses:
            abort.check()
            sock = socket.socket(family, socket.SOCK_STREAM)
            try:
                abort.attach(sock)
                sock.settimeout(timeout)
                sock.connect(sockaddr)
                return sock
            except OSError as error:
                sock.close()
                failure = error
        raise failure or OSError("no address")

    return create


class _HTTPConnection(http.client.HTTPConnection):
    def __init__(self, target, timeout, abort):
        super().__init__(target.connect_host, target.port, timeout=timeout)
        self._create_connection = _pinned(target.addresses, abort)
        self._abort = abort

    def connect(self):
        super().connect()
        self._abort.attach(self.sock)


class _HTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, target, timeout, abort):
        super().__init__(target.connect_host, target.port, timeout=timeout, context=_TLS)
        self._create_connection = _pinned(target.addresses, abort)
        self._abort = abort

    def connect(self):
        super().connect()
        self._abort.attach(self.sock)


def _open(target, method, headers, body, timeout, abort):
    """Sends one request to one checked target; returns (connection, response)."""
    kind = _HTTPSConnection if target.scheme == "https" else _HTTPConnection
    connection = kind(target, timeout, abort)
    try:
        abort.check()
        connection.request(method, target.path, body=body, headers=headers)
        response = connection.getresponse()
        abort.check()
    except BaseException:
        connection.close()
        raise
    return connection, response


def _redecode(value):
    """http.client reads header bytes as Latin-1; most servers meant UTF-8."""
    try:
        return value.encode("latin-1").decode("utf-8")
    except UnicodeError:
        return value


def _timeout_seconds(value):
    """iOS: options.timeout (a number or a numeric string), default 15000, held to 1000–30000 ms."""
    number = math.nan
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
    elif isinstance(value, str) and value == value.strip():
        try:
            number = float(value)
        except ValueError:
            pass
    if math.isnan(number):
        number = 15000.0
    return min(max(number, 1000.0), 30000.0) / 1000


def _request_headers(options):
    """iOS LXNetworkClient.fetch: at most 32, string values, no CR/LF, none of the forbidden names, 16 KiB in all."""
    raw = options.get("headers")
    if not isinstance(raw, dict):
        return {}
    if len(raw) > MAX_HEADERS:
        raise NetError("request_rejected")
    headers = {}
    total = 0
    for name, value in raw.items():
        if not isinstance(value, str) or "\r" in name or "\n" in name or "\r" in value or "\n" in value:
            raise NetError("request_rejected")
        if name.lower() in FORBIDDEN_HEADERS or not _TOKEN.fullmatch(name):
            raise NetError("request_rejected")
        total += _utf8_length(name) + _utf8_length(value)
        if total > MAX_HEADER_BYTES:
            raise NetError("request_rejected")
        headers[name] = value
    return headers


def _header(headers, name):
    lower = name.lower()
    return next((value for key, value in headers.items() if key.lower() == lower), None)


def _is_json_type(content_type):
    if content_type is None:
        return False
    first = next((part for part in content_type.split(";", 1) if part), "")
    return first.strip().lower() == "application/json"


def _form_scalar(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 2 ** 63:
            return str(int(value))
        return repr(value)
    return None


def _component(text):
    # Exactly what JavaScript's encodeURIComponent leaves alone; '-_.~' are always kept by quote.
    return urllib.parse.quote(text, safe="!*'()")


def _form_body(form):
    pairs = []
    for key in sorted(form):
        text = _form_scalar(form[key])
        if text is None:
            raise NetError("request_rejected")
        pairs.append(_component(key) + "=" + _component(text))
    return "&".join(pairs).encode("ascii")


def _prepare(method, options):
    """iOS LXNetworkRequestEncoder.prepare: default headers, the content type, the body."""
    headers = _request_headers(options)
    if _header(headers, "Accept") is None:
        headers["Accept"] = DEFAULT_ACCEPT
    if _header(headers, "User-Agent") is None:
        headers["User-Agent"] = DEFAULT_USER_AGENT
    has_body, has_form, has_form_data = "body" in options, "form" in options, "formData" in options
    if method == "POST" and _header(headers, "Content-Type") is None:
        if has_form:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        elif has_form_data:
            raise NetError("request_rejected")
        elif has_body:
            headers["Content-Type"] = "application/json"
    if isinstance(options.get("form"), dict):
        body = _form_body(options["form"])
    elif has_form_data:
        raise NetError("request_rejected")
    elif has_body:
        value = options["body"]
        if isinstance(value, str):
            body = value.encode("utf-8")
        elif _is_json_type(_header(headers, "Content-Type")):
            body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        else:
            raise NetError("request_rejected")
    else:
        body = None
    if body is not None and len(body) > MAX_REQUEST_BYTES:
        raise NetError("request_rejected")
    # One value per name, the last one winning, as URLRequest.setValue does; we decode gzip and deflate ourselves.
    final = {}
    for name, value in headers.items():
        if name.lower() == "accept-encoding":
            continue
        for existing in [key for key in final if key.lower() == name.lower()]:
            del final[existing]
        final[name] = value.encode("utf-8")
    final["Accept-Encoding"] = b"gzip, deflate"
    return final, body


class _Decoder:
    """gzip and deflate, the way URLSession hands back a decoded body; deflate may come without its zlib header."""

    def __init__(self, encoding):
        self._raw_fallback = encoding == "deflate"
        self._inflater = zlib.decompressobj(47)
        self._started = False

    def feed(self, chunk, limit):
        try:
            data = self._inflater.decompress(chunk, limit)
        except zlib.error:
            if not self._raw_fallback or self._started:
                raise
            self._inflater = zlib.decompressobj(-15)
            data = self._inflater.decompress(chunk, limit)
        self._started = True
        return data


def _read_body(response):
    encoding = (response.getheader("Content-Encoding") or "").strip().lower()
    decoder = _Decoder(encoding) if encoding in ("gzip", "x-gzip", "deflate") else None
    body = bytearray()
    raw = 0
    while True:
        chunk = response.read(_CHUNK)
        if not chunk:
            break
        raw += len(chunk)
        if raw > MAX_RESPONSE_BYTES:
            raise NetError("response_too_large")
        body += decoder.feed(chunk, MAX_RESPONSE_BYTES + 1 - len(body)) if decoder else chunk
        if len(body) > MAX_RESPONSE_BYTES:
            raise NetError("response_too_large")
    return bytes(body)


def _response_headers(response):
    """iOS: no Set-Cookie or Authorization, repeated names joined with ', ', at most 32, each cut to 2048."""
    merged = {}
    names = {}
    for name, value in response.getheaders():
        lower = name.lower()
        if lower in HIDDEN_RESPONSE_HEADERS:
            continue
        value = _redecode(value)
        if lower in names:
            merged[names[lower]] += ", " + value
        elif len(merged) < MAX_RESPONSE_HEADERS:
            names[lower] = name
            merged[name] = value
    return {name: value[:MAX_RESPONSE_HEADER_CHARS] for name, value in merged.items()}


def _exchange(target, url, method, headers, body, timeout, allow_http, allow_private, abort):
    """Follows at most 6 redirects; each new address is checked like the first."""
    current = url
    for hop in range(MAX_REDIRECTS + 1):
        if hop:
            target = _target(current, allow_http=allow_http, allow_private=allow_private)
        connection, response = _open(target, method, headers, body, timeout, abort)
        try:
            location = response.getheader("Location")
            if response.status in REDIRECT_STATUSES and location is not None:
                if hop == MAX_REDIRECTS:
                    raise NetError("request_rejected")
                current = urllib.parse.urljoin(current, _redecode(location).strip())
                if (response.status == 303 and method != "HEAD") or (response.status in (301, 302) and method == "POST"):
                    method, body = "GET", None
                    headers = {name: value for name, value in headers.items() if name.lower() != "content-type"}
                continue
            return response.status, _response_headers(response), _read_body(response)
        finally:
            connection.close()
    raise NetError("request_rejected")


def _reply(status, headers, body, binary):
    """iOS LXMusicWebWorkerNetworkHandler.reply: what the script gets back."""
    value = {"statusCode": status, "statusMessage": "", "headers": headers, "bytes": len(body)}
    if binary:
        value["body"] = {"type": "Buffer", "data": list(body)}
        text = ""
    else:
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            text = ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")), text


def perform(url, options, *, allow_private=False, abort=None):
    """One lx.request, by iOS perform and fetch.

    Returns a Reply: the error code ("" when there is a response, whatever its
    status), the response JSON and body text for the script, and the host name
    and status code for the caller's log line.
    """
    abort = abort or Abort()
    host = None
    try:
        if not isinstance(url, str) or _utf8_length(url) > MAX_URL_BYTES or not isinstance(options, dict):
            raise NetError("request_rejected")
        try:
            options_bytes = _utf8_length(json.dumps(options, ensure_ascii=False, separators=(",", ":")))
            scheme = urllib.parse.urlsplit(url).scheme.lower()
        except (TypeError, ValueError):
            raise NetError("request_rejected") from None
        if options_bytes > MAX_OPTIONS_BYTES:
            raise NetError("request_rejected")
        method = options["method"].upper() if isinstance(options.get("method"), str) else "GET"
        # iOS sends plain http only as a bodiless GET or HEAD.
        allow_http = scheme == "http"
        if allow_http and (method not in ("GET", "HEAD") or any(key in options for key in ("body", "form", "formData"))):
            raise NetError("request_rejected")
        target = _target(url, allow_http=allow_http, allow_private=allow_private)
        host = target.host
        if method not in METHODS:
            raise NetError("request_rejected")
        timeout = _timeout_seconds(options.get("timeout"))
        try:
            headers, body = _prepare(method, options)
        except UnicodeError:
            raise NetError("request_rejected") from None
        deadline = threading.Timer(min(timeout + 5, 35), abort.abort, ("timeout",))
        deadline.daemon = True
        deadline.start()
        try:
            status, response_headers, data = _exchange(
                target, url, method, headers, body, timeout, allow_http, allow_private, abort,
            )
        except NetError as error:
            raise NetError(abort.reason or error.code) from None
        except ValueError:
            raise NetError(abort.reason or "request_rejected") from None
        except TimeoutError:
            raise NetError(abort.reason or "timeout") from None
        except (OSError, http.client.HTTPException, zlib.error):
            raise NetError(abort.reason or "unavailable") from None
        finally:
            deadline.cancel()
        response_json, text = _reply(status, response_headers, data, options.get("binary") is True)
        return Reply("", response_json, text, host, status)
    except NetError as error:
        return Reply(error.code, "{}", "", host, None)


def media_url(url, *, allow_private=False):
    """iOS canonicalMediaURL for a musicUrl answer: http or https, its host and every address public.

    Returns the URL unchanged; raises NetError otherwise.
    """
    _target(url, allow_http=True, allow_private=allow_private)
    return url


def is_unplayable(url):
    """iOS isUnplayableMediaFormat: by the extension of the decoded path."""
    try:
        path = urllib.parse.unquote(urllib.parse.urlsplit(url).path)
    except ValueError:
        return False
    name = path.rstrip("/").rsplit("/", 1)[-1]
    return posixpath.splitext(name)[1][1:].lower() in UNPLAYABLE_EXTENSIONS


def host_of(url):
    """The host name alone, for a log line."""
    try:
        return (urllib.parse.urlsplit(url).hostname or "unknown")[:253]
    except (TypeError, ValueError):
        return "unknown"


def open_media(url, headers, *, allow_private=False, timeout=15.0, abort=None):
    """For the relay: GET the audio, checking every hop's addresses.

    Returns (connection, response); the caller closes the connection.
    """
    abort = abort or Abort()
    current = url
    for hop in range(MAX_REDIRECTS + 1):
        target = _target(current, allow_http=True, allow_private=allow_private)
        connection, response = _open(target, "GET", headers, None, timeout, abort)
        location = response.getheader("Location")
        if response.status not in REDIRECT_STATUSES or location is None:
            return connection, response
        connection.close()
        current = urllib.parse.urljoin(current, _redecode(location).strip())
    raise NetError("request_rejected")
