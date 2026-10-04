"""Card 51: the LX source runner, the resolver and their endpoints.

Nothing here leaves 127.0.0.1: a local server stands in for the internet. The tests that run the
real Node runner are skipped where Node is not installed.
"""

import base64
import contextlib
import datetime
import functools
import hashlib
import http.client
import http.server
import ipaddress
import json
import logging
import os
import re
import shutil
import signal
import stat
import threading
import time
import urllib.parse

import pytest

from analyzer import serve
from analyzer.access import AccessSettings
from analyzer.downloads import DownloadError
from analyzer.log import LOGGER
from analyzer.lxhost import Runner, RunnerError
from analyzer.resolver import (
    CACHE_SECONDS, COOLDOWN_SECONDS, MAX_STREAMS, OWNER, REPORTED_SECONDS, SHANGHAI, TICKET_SECONDS,
    ResolveError, Resolver, SourceLedger, Transports,
)
from analyzer.serve import _Handler, make_server
from analyzer.sources import SourceError, SourceStore

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node is not installed here")
TIERS = ("128k", "320k", "flac")
TOKEN = "xiyue-test-token-0123456789abcdef"


class _Clock:
    def __init__(self, now=None):
        self.now = now if now is not None else datetime.datetime(2026, 10, 4, 12, 0, tzinfo=SHANGHAI).timestamp()

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class _Lines(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


@pytest.fixture
def logged():
    handler = _Lines()
    level = LOGGER.level
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    try:
        yield handler.lines
    finally:
        LOGGER.removeHandler(handler)
        LOGGER.setLevel(level)


def _settle(check, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not check():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


class _Upstream:
    """The internet as far as these tests go: what the scripts' lx.request asks, and the audio."""

    def __init__(self):
        self.answer = lambda tag, platform, quality: f"/media/{tag}-{platform}-{quality}.mp3"
        self.media = {}
        self.requests = []
        self.resolved = []
        self.active = 0
        self.arrived = threading.Event()
        self.release = threading.Event()
        self._lock = threading.Lock()
        upstream = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                with upstream._lock:
                    upstream.active += 1
                    upstream.requests.append((self.path, {key.lower(): value for key, value in self.headers.items()}))
                try:
                    url = urllib.parse.urlsplit(self.path)
                    if url.path == "/resolve":
                        upstream._resolve(self, urllib.parse.parse_qs(url.query))
                    else:
                        upstream._media(self, url.path.removeprefix("/media/"))
                except OSError:
                    pass
                finally:
                    with upstream._lock:
                        upstream.active -= 1

            def log_message(self, format, *args):
                return

        class Server(http.server.ThreadingHTTPServer):
            def handle_error(self, request, client_address):
                # A relay that hangs up mid-file is what some tests are about.
                return

        self._server = Server(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self._server.server_address[1]}"

    def url(self, path):
        return self.base + path

    def _resolve(self, handler, query):
        tag, platform, quality, songmid = (query.get(name, [""])[0] for name in ("tag", "source", "quality", "songmid"))
        with self._lock:
            self.resolved.append((tag, platform, quality, songmid))
        path = self.answer(tag, platform, quality)
        if path == "hold":
            self.arrived.set()
            self.release.wait(30)
            path = None
        status, body = (500, b"{}") if path is None else (200, json.dumps({"url": self.base + path}).encode())
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def _media(self, handler, name):
        if name not in self.media:
            handler.send_error(404)
            return
        content_type, body = self.media[name]
        # An int body is that many zero bytes, sent as it goes.
        total = body if isinstance(body, int) else len(body)
        start, end, status = 0, total - 1, 200
        wanted = re.fullmatch(r"bytes=(\d+)-(\d*)", handler.headers.get("Range", ""))
        if wanted:
            start, status = int(wanted.group(1)), 206
            if wanted.group(2):
                end = min(int(wanted.group(2)), total - 1)
        handler.send_response(status)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(end - start + 1))
        handler.send_header("Accept-Ranges", "bytes")
        handler.send_header("Set-Cookie", "upstream=private")
        if status == 206:
            handler.send_header("Content-Range", f"bytes {start}-{end}/{total}")
        handler.end_headers()
        if isinstance(body, bytes):
            handler.wfile.write(body[start:end + 1])
            return
        left, block = end - start + 1, bytes(64 * 1024)
        while left > 0:
            piece = block[:min(left, len(block))]
            handler.wfile.write(piece)
            left -= len(piece)

    def close(self):
        self.release.set()
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def upstream():
    server = _Upstream()
    try:
        yield server
    finally:
        server.close()


class _StubRunner:
    """Node's place in the tests that are not about Node: answers musicUrl from a function."""

    running = True

    def __init__(self):
        self.declared = {}
        self.names = {}
        self.calls = []
        self.answer = lambda name, platform, tier, info: _url(platform, info, tier)
        self._loaded = set()

    def loaded(self):
        return frozenset(self._loaded)

    def load(self, source_id, script, meta):
        self.names[source_id] = meta["name"]
        self._loaded.add(source_id)
        return {platform: list(tiers) for platform, tiers in self.declared[meta["name"]].items()}

    def unload(self, source_id):
        self._loaded.discard(source_id)

    def call(self, source_id, platform, quality, music_info):
        name = self.names[source_id]
        self.calls.append((name, platform, quality))
        if source_id not in self._loaded:
            raise RunnerError("unavailable")
        answer = self.answer(name, platform, quality, music_info)
        if isinstance(answer, Exception):
            raise answer
        return answer


class _Downloads:
    def __init__(self, error=None):
        self.jobs, self.error = [], error

    def available(self):
        return True

    def submit(self, job):
        if self.error is not None:
            raise self.error
        self.jobs.append(job)
        return f"job-{len(self.jobs)}"


def _url(platform, info, tier, extension="mp3"):
    return f"http://127.0.0.1:9/{platform}/{urllib.parse.quote(str(info['songmid']))}/{tier}.{extension}"


def _song(platform, songmid, quality="320k", exact=False):
    info = {
        "name": "测试歌", "singer": "测试", "source": platform, "interval": "03:00", "albumName": "",
        "albumId": "", "img": "", "types": [], "_types": {}, "typeUrl": {}, "songmid": songmid,
    }
    if platform == "tx":
        info.update(songId=1, strMediaMid="m" + songmid)
    if platform == "kg":
        info["hash"] = "0" * 32
    return {"platform": platform, "musicInfo": info, "quality": quality, "exact": exact}


def _stubbed(tmp_path, clock=None):
    runner = _StubRunner()
    store = SourceStore(tmp_path, runner)
    return runner, store, Resolver(store, runner, tmp_path, clock=clock or _Clock(), allow_private=True)


def _add(store, runner, name, platforms=("kw", "kg"), quota=None):
    runner.declared[name] = platforms if isinstance(platforms, dict) else {platform: TIERS for platform in platforms}
    source_id = store.add(f"/*\n * @name {name}\n * @version 1.0.0\n */\n")["id"]
    if quota is not None:
        store.update(source_id, {"dailyQuota": quota})
    return source_id


def _today(resolver):
    """(musicUrl calls today, {account: (downloads, plays)})."""
    today = resolver.usage()["days"][0]
    return today["used"], {row["account"]: (row["download"], row["play"]) for row in today["accounts"]}


@contextlib.contextmanager
def _container(tmp_path, store, resolver, access=None, downloads=None):
    server = make_server(
        tmp_path, 0, data=tmp_path, downloads=downloads, access=access, sources=store, resolver=resolver,
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


def _call(port, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        payload = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        sent = {"Content-Type": "application/json"} if payload is not None else {}
        connection.request(method, path, payload, {**sent, **(headers or {})})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


# The real runner.

def _script(base, tag, platforms=("kw", "kg")):
    sources = {
        platform: {"name": platform, "type": "music", "actions": ["musicUrl"], "qualitys": list(TIERS)}
        for platform in platforms
    }
    return f"""/*
 * @name 测试源 {tag}
 * @version 1.0.0
 */
const base = {json.dumps(base)}
const tag = {json.dumps(tag)}
lx.on(lx.EVENT_NAMES.request, ({{ source, action, info }}) => new Promise((resolve, reject) => {{
  if (action !== 'musicUrl') return reject(new Error('unsupported'))
  const query = ['tag=' + encodeURIComponent(tag), 'source=' + source, 'quality=' + info.type,
    'songmid=' + encodeURIComponent(String(info.musicInfo.songmid))].join('&')
  lx.request(base + '/resolve?' + query, {{ method: 'GET' }}, (error, response, body) => {{
    if (error || response.statusCode !== 200 || body == null || typeof body.url !== 'string') return reject(new Error('failed'))
    resolve(body.url)
  }})
}}))
lx.send(lx.EVENT_NAMES.inited, {{ status: true, openDevTools: false, sources: {json.dumps(sources)} }})
"""


SILENT_SCRIPT = """/*
 * @name 不初始化
 * @version 1.0.0
 */
lx.on(lx.EVENT_NAMES.request, () => Promise.resolve('https://x.invalid/a.mp3'))
"""

THROWING_SCRIPT = """/*
 * @name 一加载就报错
 * @version 1.0.0
 */
throw new Error('boom')
"""


@pytest.fixture
def node_runner():
    runners = []

    def make(start=True, **options):
        runner = Runner(node=NODE, allow_private=True, **options)
        runners.append(runner)
        if start:
            _start(runner)
        return runner

    try:
        yield make
    finally:
        for runner in runners:
            runner.close()


def _start(runner):
    if not runner.start():
        pytest.skip("this Node has no permission model")


@needs_node
def test_runner_load_reads_declared_sources_and_fails_without_init(tmp_path, node_runner, upstream):
    runner = node_runner(load_timeout=1.0)
    store = SourceStore(tmp_path, runner)

    entry = store.add(_script(upstream.base, "A"))
    assert (entry["name"], entry["enabled"], entry["loadError"]) == ("测试源 A", True, "")
    assert entry["platforms"] == {"kw": list(TIERS), "kg": list(TIERS)}
    assert entry["id"] in runner.loaded()

    started = time.monotonic()
    silent = store.add(SILENT_SCRIPT)
    assert 0.9 <= time.monotonic() - started < 4
    assert (silent["enabled"], silent["loadError"]) == (False, "initialization_timeout")
    assert silent["id"] not in runner.loaded()
    # Kept, switched off, for the panel to show why.
    assert (tmp_path / "sources" / f"{silent['id']}.js").read_text(encoding="utf-8") == SILENT_SCRIPT

    throwing = store.add(THROWING_SCRIPT)
    assert (throwing["enabled"], throwing["loadError"]) == (False, "script_failed")
    with pytest.raises(SourceError) as error:
        store.add(_script(upstream.base, "A"))
    assert (error.value.code, error.value.status) == ("duplicate", 409)

    directory = tmp_path / "sources"
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert {stat.S_IMODE(path.stat().st_mode) for path in directory.iterdir()} == {0o600}
    assert "lx.on" not in (directory / "index.json").read_text(encoding="utf-8")
    assert "lx.on" not in json.dumps(store.list(), ensure_ascii=False)


# What the natives give a script, worked out again in Python: AES-128 (FIPS-197) and textbook RSA.

def _gmul(a, b):
    product = 0
    while b:
        if b & 1:
            product ^= a
        a = (a << 1) ^ 0x11B if a & 0x80 else a << 1
        b >>= 1
    return product


@functools.cache
def _sbox():
    inverse = [0] + [next(b for b in range(1, 256) if _gmul(a, b) == 1) for a in range(1, 256)]

    def rotate(value, shift):
        return ((value << shift) | (value >> (8 - shift))) & 0xFF

    return [value ^ rotate(value, 1) ^ rotate(value, 2) ^ rotate(value, 3) ^ rotate(value, 4) ^ 0x63 for value in inverse]


def _round_keys(key):
    sbox = _sbox()
    words = [list(key[index:index + 4]) for index in range(0, 16, 4)]
    rcon = 1
    for index in range(4, 44):
        word = list(words[-1])
        if index % 4 == 0:
            word = [sbox[value] for value in word[1:] + word[:1]]
            word[0] ^= rcon
            rcon = _gmul(rcon, 2)
        words.append([left ^ right for left, right in zip(words[-4], word)])
    return [sum(words[index:index + 4], []) for index in range(0, 44, 4)]


def _encrypt_block(block, keys):
    sbox = _sbox()
    state = [value ^ key for value, key in zip(block, keys[0])]
    for number in range(1, 11):
        state = [sbox[value] for value in state]
        state = [state[row + 4 * ((column + row) % 4)] for column in range(4) for row in range(4)]
        if number < 10:
            mixed = []
            for column in range(4):
                a = state[4 * column:4 * column + 4]
                mixed += [
                    _gmul(a[0], 2) ^ _gmul(a[1], 3) ^ a[2] ^ a[3],
                    a[0] ^ _gmul(a[1], 2) ^ _gmul(a[2], 3) ^ a[3],
                    a[0] ^ a[1] ^ _gmul(a[2], 2) ^ _gmul(a[3], 3),
                    _gmul(a[0], 3) ^ a[1] ^ a[2] ^ _gmul(a[3], 2),
                ]
            state = mixed
        state = [value ^ key for value, key in zip(state, keys[number])]
    return bytes(state)


def _aes(data, key, iv=None):
    """AES-128 with PKCS#7 padding: CBC with an IV, ECB without."""
    keys = _round_keys(key)
    padding = 16 - len(data) % 16
    data += bytes([padding]) * padding
    out, previous = b"", iv
    for index in range(0, len(data), 16):
        block = data[index:index + 16]
        if iv is not None:
            block = bytes(left ^ right for left, right in zip(block, previous))
        previous = _encrypt_block(block, keys)
        out += previous
    return out


def _der(tag, body):
    if len(body) < 0x80:
        return bytes([tag, len(body)]) + body
    size = (len(body).bit_length() + 7) // 8
    return bytes([tag, 0x80 | size]) + len(body).to_bytes(size, "big") + body


def _der_integer(value):
    return _der(0x02, value.to_bytes(value.bit_length() // 8 + 1, "big"))


def _rsa_public_key():
    """A made-up 1024-bit modulus (odd, top bit set) in base64 SPKI between PEM lines."""
    digest = hashlib.sha512(b"xiyue-rsa-1").digest() + hashlib.sha512(b"xiyue-rsa-2").digest()
    modulus = int.from_bytes(digest, "big") | (1 << 1023) | 1
    algorithm = _der(0x30, _der(0x06, bytes.fromhex("2a864886f70d010101")) + b"\x05\x00")
    public = _der(0x30, _der_integer(modulus) + _der_integer(65537))
    encoded = base64.b64encode(_der(0x30, algorithm + _der(0x03, b"\x00" + public))).decode("ascii")
    lines = "\n".join(encoded[index:index + 64] for index in range(0, len(encoded), 64))
    return modulus, f"-----BEGIN PUBLIC KEY-----\n{lines}\n-----END PUBLIC KEY-----\n"


AES_TEXT = "汐乐 aes 测试 0123456789"
AES_KEY = "k-0123456789abcd"
AES_IV = "i-0123456789abcd"
RSA_TEXT = "汐乐 rsa"
MD5_TEXT = "汐乐 md5 ✓"
B64_TEXT = "汐乐 b64"
RAW_BYTES = bytes(range(0, 256, 7))


def _crypto_script(public_key):
    def literal(value):
        return json.dumps(value)

    return f"""/*
 * @name 加解密
 * @version 1.0.0
 */
const cryptoUtils = lx.utils.crypto
const bufferUtils = lx.utils.buffer
const hex = data => bufferUtils.bufToString(data, 'hex')
lx.on(lx.EVENT_NAMES.request, () => {{
  const values = {{
    cbc: hex(cryptoUtils.aesEncrypt({literal(AES_TEXT)}, 'aes-128-cbc', {literal(AES_KEY)}, {literal(AES_IV)})),
    ecb: hex(cryptoUtils.aesEncrypt(bufferUtils.from({literal(AES_TEXT)}), 'aes-128-ecb', {literal(AES_KEY)}, '')),
    rsa: hex(cryptoUtils.rsaEncrypt(bufferUtils.from({literal(RSA_TEXT)}), {literal(public_key)})),
    md5: cryptoUtils.md5({literal(MD5_TEXT)}),
    b64: bufferUtils.bufToString(bufferUtils.from({literal(B64_TEXT)}), 'base64'),
    buf: hex(bufferUtils.from({literal(base64.b64encode(RAW_BYTES).decode())}, 'base64')),
  }}
  return Promise.resolve('https://x.invalid/?' + Object.keys(values).map(name => name + '=' + encodeURIComponent(values[name])).join('&'))
}})
lx.send(lx.EVENT_NAMES.inited, {{ status: true, sources: {{ kw: {{ type: 'music', actions: ['musicUrl'], qualitys: ['128k'] }} }} }})
"""


@needs_node
def test_runner_crypto_natives_match_python(tmp_path, node_runner):
    assert _encrypt_block(
        bytes.fromhex("00112233445566778899aabbccddeeff"), _round_keys(bytes.fromhex("000102030405060708090a0b0c0d0e0f")),
    ).hex() == "69c4e0d86a7b0430d8cdb78070b4c55a"
    modulus, public_key = _rsa_public_key()
    runner = node_runner()
    entry = SourceStore(tmp_path, runner).add(_crypto_script(public_key))
    assert entry["loadError"] == ""

    # The runner hands the answer back unchecked, so x.invalid is never looked up.
    answer = runner.call(entry["id"], "kw", "128k", {"songmid": "1"})
    got = {name: values[0] for name, values in urllib.parse.parse_qs(urllib.parse.urlsplit(answer).query).items()}

    key, text = AES_KEY.encode(), AES_TEXT.encode()
    message = int.from_bytes(RSA_TEXT.encode(), "big")
    assert got == {
        "cbc": _aes(text, key, AES_IV.encode()).hex(),
        "ecb": _aes(text, key).hex(),
        "rsa": pow(message, 65537, modulus).to_bytes(128, "big").hex(),
        "md5": hashlib.md5(MD5_TEXT.encode()).hexdigest(),
        "b64": base64.b64encode(B64_TEXT.encode()).decode(),
        "buf": RAW_BYTES.hex(),
    }


@needs_node
def test_runner_steps_down_past_an_encrypted_tier_unless_exact(tmp_path, node_runner, upstream, logged):
    runner = node_runner()
    store = SourceStore(tmp_path, runner)
    resolver = Resolver(store, runner, tmp_path, clock=_Clock(), allow_private=True)
    store.add(_script(upstream.base, "A"))
    upstream.answer = lambda tag, platform, quality: f"/media/{quality}." + ("mflac" if quality == "flac" else "mp3")

    found = resolver.resolve(_song("kw", "1001", "flac"), OWNER)
    assert (found["quality"], found["transport"], found["url"]) == ("320k", "direct", upstream.url("/media/320k.mp3"))
    assert [item[1:] for item in upstream.resolved] == [("kw", "flac", "1001"), ("kw", "320k", "1001")]

    upstream.resolved.clear()
    with pytest.raises(ResolveError) as error:
        resolver.resolve(_song("kw", "1002", "flac", exact=True), OWNER)
    assert (error.value.code, error.value.status) == ("resolve_failed", 502)
    assert [item[1:] for item in upstream.resolved] == [("kw", "flac", "1002")]
    # Every musicUrl counts for the source; the account only for the song it got.
    assert _today(resolver) == (3, {OWNER: (0, 1)})
    text = "\n".join(logged)
    assert "/media/" not in text and "lx.on" not in text


@needs_node
def test_runner_source_from_upload_to_relayed_audio(tmp_path, node_runner, upstream, monkeypatch):
    monkeypatch.setattr(_Handler, "_is_trusted", lambda self: True)
    runner = node_runner()
    store = SourceStore(tmp_path, runner)
    resolver = Resolver(store, runner, tmp_path, allow_private=True)
    audio = bytes(range(256)) * 64
    upstream.media["song.mp3"] = ("audio/mpeg", audio)
    upstream.answer = lambda tag, platform, quality: "/media/song.mp3"

    with _container(tmp_path, store, resolver) as port:
        status, _, body = _call(port, "POST", "/api/sources", {"script": _script(upstream.base, "A")})
        assert status == 200 and b"lx.on" not in body
        added = json.loads(body)
        assert added["enabled"] is True
        status, _, body = _call(port, "POST", "/api/source/resolve", _song("kg", "2001", "320k"))
        assert status == 200 and b"/media/" not in body
        answer = json.loads(body)
        assert (answer["transport"], answer["quality"], answer["sourceName"]) == ("relay", "320k", "测试源 A")
        assert answer["sourceID"] == added["id"] and "url" not in answer
        status, headers, data = _call(port, "GET", answer["path"], headers={"Range": "bytes=10-19"})
    assert (status, data) == (206, audio[10:20])
    assert headers["Content-Range"] == f"bytes 10-19/{len(audio)}"


@needs_node
def test_runner_restarts_and_reloads_after_node_is_killed(tmp_path, node_runner, upstream, logged):
    runner = node_runner(start=False)
    store = SourceStore(tmp_path, runner)
    runner.on_ready = store.reload_all
    _start(runner)
    entry = store.add(_script(upstream.base, "A"))
    first = runner.pid

    upstream.answer = lambda tag, platform, quality: "hold"
    errors = []

    def call():
        try:
            runner.call(entry["id"], "kw", "320k", {"songmid": "1"})
        except RunnerError as error:
            errors.append(error)

    waiting = threading.Thread(target=call)
    waiting.start()
    assert upstream.arrived.wait(10)
    os.kill(first, signal.SIGKILL)
    waiting.join(10)
    assert [(error.code, error.crashed) for error in errors] == [("runner_crashed", True)]

    assert _settle(lambda: runner.pid not in (None, first) and entry["id"] in runner.loaded(), 20)
    upstream.answer = lambda tag, platform, quality: "/media/back.mp3"
    assert runner.call(entry["id"], "kw", "320k", {"songmid": "2"}) == upstream.url("/media/back.mp3")
    assert store.get(entry["id"])["enabled"] is True
    text = "\n".join(logged)
    assert "LXRUNNER exit" in text and "LXRUNNER started generation=2" in text


# Picking a source.

def test_resolver_rotation_skips_spent_quota_and_owner_tries_every_source(tmp_path):
    runner, store, resolver = _stubbed(tmp_path)
    _add(store, runner, "A", quota=1)
    b = _add(store, runner, "B")

    assert resolver.resolve(_song("kw", "1"), "family-test")["sourceName"] == "A"
    # A has spent its quota: the family goes to B, and the owner asks B before A.
    assert resolver.resolve(_song("kw", "2"), "family-test")["sourceName"] == "B"
    assert resolver.resolve(_song("kw", "3"), OWNER)["sourceName"] == "B"

    runner.calls.clear()
    runner.answer = lambda name, platform, tier, info: RunnerError("script_failed") if name == "B" else _url(platform, info, tier)
    assert resolver.resolve(_song("kw", "4", "flac"), OWNER)["sourceName"] == "A"
    assert runner.calls == [("B", "kw", "flac"), ("B", "kw", "320k"), ("B", "kw", "128k"), ("A", "kw", "flac")]

    # With every quota spent the family is turned away without a call; the owner never is.
    store.update(b, {"dailyQuota": resolver.ledger.used_today()[b]})
    runner.calls.clear()
    with pytest.raises(ResolveError) as error:
        resolver.resolve(_song("kw", "5"), "family-test")
    assert (error.value.code, error.value.status, runner.calls) == ("sources_exhausted", 429, [])
    assert resolver.resolve(_song("kw", "5"), OWNER)["sourceName"] == "A"


def test_resolver_cools_a_source_down_after_three_failures_and_asks_it_last(tmp_path, logged):
    clock = _Clock()
    runner, store, resolver = _stubbed(tmp_path, clock)
    a = _add(store, runner, "A", platforms={"kw": ("128k",), "kg": TIERS})
    _add(store, runner, "B", platforms=("kw",))
    failing = {("A", "kg")}
    runner.answer = lambda name, platform, tier, info: (
        RunnerError("script_failed") if (name, platform) in failing else _url(platform, info, tier)
    )

    for number in range(3):
        with pytest.raises(ResolveError) as error:
            resolver.resolve(_song("kg", f"k{number}", exact=True), OWNER)
        assert error.value.code == "resolve_failed"
    rows = {row["name"]: row for row in resolver.panel_sources()["sources"]}
    assert (rows["A"]["cooldownUntil"], rows["A"]["failures"], rows["A"]["lastError"]) == (
        int(clock() + COOLDOWN_SECONDS), 3, "script_failed",
    )
    assert rows["B"]["cooldownUntil"] is None
    assert f"SOURCE-COOLDOWN source={a} minutes=10" in logged

    # B fails four times (only B has flac) with a success between, so it does not cool down
    # yet has more recent failures than A.
    for number, fails in enumerate((True, True, False, True, True)):
        (failing.add if fails else failing.discard)(("B", "kw"))
        try:
            resolver.resolve(_song("kw", f"f{number}", "flac", exact=True), OWNER)
        except ResolveError:
            assert fails
    failing.discard(("B", "kw"))
    rows = {row["name"]: row for row in resolver.panel_sources()["sources"]}
    assert (rows["B"]["failures"], rows["B"]["cooldownUntil"]) == (4, None)

    # A comes first in the panel and has fewer failures, yet while it cools B is asked first.
    runner.calls.clear()
    assert resolver.resolve(_song("kw", "w1", "128k"), OWNER)["sourceName"] == "B"
    assert runner.calls == [("B", "kw", "128k")]
    clock.advance(COOLDOWN_SECONDS)
    runner.calls.clear()
    assert resolver.resolve(_song("kw", "w2", "128k"), OWNER)["sourceName"] == "A"
    assert runner.calls == [("A", "kw", "128k")]


# Counting.

def test_resolver_counts_the_owner_but_never_refuses_it(tmp_path):
    runner, store, resolver = _stubbed(tmp_path)
    source_id = _add(store, runner, "A")
    downloads = _Downloads()
    for number in range(1001):
        resolver.resolve(_song("kw", f"p{number}"), OWNER)
    for number in range(201):
        resolver.download(_song("kw", f"d{number}"), OWNER, downloads)
    assert store.get(source_id)["dailyQuota"] == 1000
    assert resolver.ledger.used_today() == {source_id: 1202}
    assert _today(resolver) == (1202, {OWNER: (201, 1001)})
    assert len(downloads.jobs) == 201


def test_resolver_stops_a_family_account_at_its_limits_before_any_source_call(tmp_path):
    runner, store, resolver = _stubbed(tmp_path)
    _add(store, runner, "A")
    downloads = _Downloads()
    for number in range(300):
        resolver.resolve(_song("kw", f"p{number}"), "family-test")
    calls = len(runner.calls)
    with pytest.raises(ResolveError) as error:
        resolver.resolve(_song("kw", "p300"), "family-test")
    assert (error.value.code, error.value.status) == ("quota_exceeded", 429)
    assert len(runner.calls) == calls

    for number in range(200):
        resolver.download(_song("kw", f"d{number}"), "family-test", downloads)
    with pytest.raises(ResolveError) as error:
        resolver.download(_song("kw", "d200"), "family-test", downloads)
    assert error.value.code == "quota_exceeded"
    assert len(downloads.jobs) == 200

    # Raised in the panel, the limit lets the next play through.
    resolver.set_limits({"familyPlaysPerDay": 301})
    resolver.resolve(_song("kw", "p300"), "family-test")
    assert _today(resolver)[1] == {"family-test": (200, 301)}


def test_resolver_cache_hits_call_no_source_and_count_a_play_once(tmp_path):
    clock = _Clock()
    runner, store, resolver = _stubbed(tmp_path, clock)
    _add(store, runner, "A")
    runner.answer = lambda name, platform, tier, info: _url(platform, info, tier, "mflac" if tier == "flac" else "mp3")

    first = resolver.resolve(_song("kw", "1", "flac"), OWNER)
    assert first["quality"] == "320k"
    assert runner.calls == [("A", "kw", "flac"), ("A", "kw", "320k")]
    assert _today(resolver) == (2, {OWNER: (0, 1)})

    # The same song again: where flac landed is remembered, the play is not counted twice.
    for quality in ("flac", "320k"):
        again = resolver.resolve(_song("kw", "1", quality), OWNER)
        assert (again["quality"], again["url"]) == ("320k", first["url"])
    assert len(runner.calls) == 2 and _today(resolver) == (2, {OWNER: (0, 1)})

    # Another account's first play and any download count, still without a source call.
    resolver.resolve(_song("kw", "1", "flac"), "family-test")
    resolver.download(_song("kw", "1", "flac"), OWNER, _Downloads())
    assert len(runner.calls) == 2 and _today(resolver) == (2, {"family-test": (0, 1), OWNER: (1, 1)})

    clock.advance(CACHE_SECONDS)
    resolver.resolve(_song("kw", "1", "flac"), OWNER)
    assert len(runner.calls) == 4 and _today(resolver) == (4, {"family-test": (0, 1), OWNER: (1, 2)})


def test_resolver_counts_start_over_at_shanghai_midnight(tmp_path):
    clock = _Clock(datetime.datetime(2026, 10, 4, 23, 59, 30, tzinfo=SHANGHAI).timestamp())
    runner, store, resolver = _stubbed(tmp_path, clock)
    source_id = _add(store, runner, "A", quota=1)
    resolver.set_limits({"familyPlaysPerDay": 1})
    resolver.resolve(_song("kw", "1"), "family-test")
    with pytest.raises(ResolveError) as error:
        resolver.resolve(_song("kw", "2"), "family-test")
    assert error.value.code == "quota_exceeded"

    clock.advance(60)
    assert resolver.resolve(_song("kw", "2"), "family-test")["sourceName"] == "A"
    days = resolver.usage()["days"]
    assert [(day["date"], day["used"]) for day in days[:2]] == [("2026-10-05", 1), ("2026-10-04", 1)]

    usage = tmp_path / "source-usage.json"
    assert stat.S_IMODE(usage.stat().st_mode) == 0o600
    saved = json.loads(usage.read_text(encoding="utf-8"))
    saved["days"]["2026-09-01"] = {"sources": {source_id: 5}, "accounts": {}}
    usage.write_text(json.dumps(saved), encoding="utf-8")
    reopened = SourceLedger(tmp_path, clock)
    assert reopened.used_today() == {source_id: 1}
    reopened.charge(OWNER, "play")
    # Only the last 31 days are kept.
    assert sorted(json.loads(usage.read_text(encoding="utf-8"))["days"]) == ["2026-10-04", "2026-10-05"]


# Relaying.

def test_relay_passes_range_through_and_tells_flac_by_its_head(tmp_path, upstream, monkeypatch, logged):
    monkeypatch.setattr(_Handler, "_is_trusted", lambda self: True)
    runner, store, resolver = _stubbed(tmp_path)
    _add(store, runner, "A")
    song = bytes(range(256)) * 64
    flac = b"fLaC" + bytes(range(256)) * 16
    upstream.media.update({
        "song.mp3": ("audio/mpeg", song), "qq.flac": ("audio/x-ogg", flac), "other.m4a": ("application/octet-stream", song),
    })
    files = {"1": "song.mp3", "2": "qq.flac", "3": "qq.flac", "4": "other.m4a"}
    runner.answer = lambda name, platform, tier, info: upstream.url("/media/" + files[info["songmid"]])

    with _container(tmp_path, store, resolver) as port:
        def relayed(songmid):
            status, _, body = _call(port, "POST", "/api/source/resolve", _song("kg", songmid))
            answer = json.loads(body)
            assert (status, answer["transport"], "url" in answer) == (200, "relay", False)
            return answer["path"]

        path = relayed("1")
        status, headers, data = _call(port, "GET", path, headers={"Range": "bytes=10-19"})
        assert (status, data) == (206, song[10:20])
        assert (headers["Content-Range"], headers["Content-Length"], headers["Accept-Ranges"], headers["Content-Type"]) == (
            f"bytes 10-19/{len(song)}", "10", "bytes", "audio/mpeg",
        )
        assert "Set-Cookie" not in headers
        sent = upstream.requests[-1][1]
        assert (sent["range"], sent["user-agent"], "authorization" in sent) == ("bytes=10-19", "Xiyue-LX-Relay/1", False)

        # A seek is relayed again, never resolved or counted again.
        before = (len(runner.calls), _today(resolver))
        status, _, data = _call(port, "GET", path, headers={"Range": "bytes=1000-"})
        assert (status, data) == (206, song[1000:])
        assert (len(runner.calls), _today(resolver)) == before

        # QQ's FLAC under audio/x-ogg, from the start and from past its head (which is then asked for alone).
        status, headers, data = _call(port, "GET", relayed("2"))
        assert (status, headers["Content-Type"], data) == (200, "audio/flac", flac)
        status, headers, data = _call(port, "GET", relayed("3"), headers={"Range": "bytes=8-15"})
        assert (status, headers["Content-Type"], data) == (206, "audio/flac", flac[8:16])
        assert [sent.get("range") for _, sent in upstream.requests[-2:]] == ["bytes=8-15", "bytes=0-3"]
        status, headers, _ = _call(port, "GET", relayed("4"))
        assert (status, headers["Content-Type"]) == (200, "application/octet-stream")
    text = "\n".join(logged)
    assert path.rsplit("/", 1)[1] not in text and "/media/" not in text


def test_relay_ticket_runs_out_after_ten_minutes(tmp_path, upstream, monkeypatch):
    monkeypatch.setattr(_Handler, "_is_trusted", lambda self: True)
    clock = _Clock()
    runner, store, resolver = _stubbed(tmp_path, clock)
    _add(store, runner, "A")
    upstream.media["song.mp3"] = ("audio/mpeg", b"ID3" + bytes(1000))
    runner.answer = lambda *args: upstream.url("/media/song.mp3")
    answer = resolver.resolve(_song("kg", "1"), OWNER)
    assert answer["expiresAt"] == int(clock() + TICKET_SECONDS)
    ticket = answer["path"].rsplit("/", 1)[1]

    stream = resolver.open_stream(ticket, None)
    try:
        assert (stream.status, b"".join(stream.chunks())) == (200, b"ID3" + bytes(1000))
    finally:
        stream.close()
    clock.advance(TICKET_SECONDS)
    with _container(tmp_path, store, resolver) as port:
        status, _, body = _call(port, "GET", answer["path"])
    assert (status, json.loads(body)) == (404, {"error": "not_found"})
    for wrong in (ticket, "", "x", "a" * 42, "a" * 44, "../" + "a" * 40, None):
        with pytest.raises(ResolveError) as error:
            resolver.open_stream(wrong, None)
        assert (error.value.code, error.value.status) == ("not_found", 404)


def test_relay_holds_six_streams_and_lets_go_when_the_phone_hangs_up(tmp_path, upstream, monkeypatch):
    monkeypatch.setattr(_Handler, "_is_trusted", lambda self: True)
    runner, store, resolver = _stubbed(tmp_path)
    _add(store, runner, "A")
    upstream.media["big.mp3"] = ("audio/mpeg", 64 * 1024 * 1024)
    runner.answer = lambda *args: upstream.url("/media/big.mp3")
    path = resolver.resolve(_song("kg", "1"), OWNER)["path"]

    def listen():
        """A phone part way into the song; closing what this returns hangs up."""
        phone = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        phone.request("GET", path)
        # Over HTTP/1.0 the answer holds the socket from here on.
        answer = phone.getresponse()
        assert answer.status == 200 and len(answer.read(65536)) == 65536
        return answer

    with _container(tmp_path, store, resolver) as port:
        baseline = threading.active_count()
        listening = []
        try:
            for _ in range(MAX_STREAMS):
                listening.append(listen())
            assert upstream.active == MAX_STREAMS
            status, _, body = _call(port, "GET", path)
            assert (status, json.loads(body)) == (429, {"error": "busy"})
        finally:
            for answer in listening:
                answer.close()
        assert _settle(lambda: upstream.active == 0)
        assert _settle(lambda: threading.active_count() <= baseline)
        # Every slot came back.
        listening = [listen() for _ in range(MAX_STREAMS)]
        assert upstream.active == MAX_STREAMS
        for answer in listening:
            answer.close()
        assert _settle(lambda: upstream.active == 0 and threading.active_count() <= baseline)


# The endpoints.

def test_source_endpoints_need_a_token_outside_and_the_panel_stays_home(tmp_path, upstream, monkeypatch, logged):
    monkeypatch.setattr(serve, "_auth_rejections", {})
    runner, store, resolver = _stubbed(tmp_path)
    source_id = _add(store, runner, "A")
    upstream.media["song.mp3"] = ("audio/mpeg", bytes(100))
    runner.answer = lambda *args: upstream.url("/media/song.mp3")
    path = resolver.resolve(_song("kg", "1"), OWNER)["path"]
    access = AccessSettings(None, token=TOKEN, network=ipaddress.ip_network("192.168.50.0/24"), host_ip="192.168.50.2")
    outside = {"X-Forwarded-For": "203.0.113.9"}
    signed = {**outside, "Authorization": "Bearer " + TOKEN}

    with _container(tmp_path, store, resolver, access=access) as port:
        for method, target, body in (
            ("GET", path, None),
            ("POST", "/api/source/resolve", _song("kw", "2")),
            ("GET", "/api/sources", None),
        ):
            status, _, data = _call(port, method, target, body, outside)
            assert (status, json.loads(data)) == (401, {"error": "unauthorized"}), target
        for method, target, body in (
            ("GET", "/api/sources", None),
            ("GET", "/api/source/limits", None),
            ("GET", "/api/source/usage", None),
            ("POST", "/api/sources", {"script": "/*\n * @name B\n * @version 1\n */\n"}),
            ("POST", f"/api/sources/{source_id}", {"enabled": False}),
            ("POST", f"/api/sources/{source_id}/test", None),
            ("POST", "/api/source/limits", {"familyPlaysPerDay": 1}),
            ("DELETE", f"/api/sources/{source_id}", None),
        ):
            status, _, data = _call(port, method, target, body, signed)
            assert (status, json.loads(data)) == (403, {"error": "home_only"}), target
        status, _, data = _call(port, "POST", "/api/source/resolve", _song("kw", "2"), signed)
        assert (status, json.loads(data)["transport"]) == (200, "direct")
        status, _, data = _call(port, "GET", path, None, signed)
        assert (status, data) == (200, bytes(100))
        assert "authorization" not in upstream.requests[-1][1]

    assert [entry["id"] for entry in store.list()] == [source_id] and store.get(source_id)["enabled"] is True
    assert resolver.limits()["familyPlaysPerDay"] == 300
    text = "\n".join(logged)
    assert "path=/api/source/stream/- " in text and path.rsplit("/", 1)[1] not in text and TOKEN not in text
    assert "SOURCE-PANEL rejected" in text


def test_source_panel_at_home_manages_sources_and_never_shows_a_script(tmp_path, monkeypatch):
    monkeypatch.setattr(_Handler, "_is_trusted", lambda self: True)
    runner, store, resolver = _stubbed(tmp_path)
    runner.declared["面板源"] = {"kw": TIERS, "tx": ("128k", "320k")}
    script = "/*\n * @name 面板源\n * @version 2.1.0\n */\nconst marker = 'do-not-show'\n"
    answers = []

    with _container(tmp_path, store, resolver) as port:
        def call(method, target, body=None, expected=200):
            status, _, data = _call(port, method, target, body)
            answers.append(data)
            assert status == expected, (target, data)
            return json.loads(data)

        added = call("POST", "/api/sources", {"script": script})
        assert (added["name"], added["version"], added["enabled"]) == ("面板源", "2.1.0", True)
        source_id = added["id"]
        assert call("POST", "/api/sources", {"script": script}, 409) == {"error": "duplicate"}
        tested = call("POST", f"/api/sources/{source_id}/test")
        assert [(item["platform"], item["ok"], item["quality"], item["host"]) for item in tested["results"]] == [
            ("kw", True, "320k", "127.0.0.1"), ("tx", True, "320k", "127.0.0.1"),
        ]
        assert call("POST", f"/api/sources/{source_id}", {"dailyQuota": 500})["dailyQuota"] == 500
        listed = call("GET", "/api/sources")
        (row,) = listed["sources"]
        assert listed["runner"] is True
        assert (row["used"], row["dailyQuota"], row["failures"], row["cooldownUntil"], row["lastError"]) == (
            2, 500, 0, None, "",
        )
        usage = call("GET", "/api/source/usage")
        assert (usage["quota"], usage["days"][0]["used"], len(usage["days"])) == (500, 2, 7)

        limits = call("POST", "/api/source/limits", {"familyDownloadsPerDay": 150, "transports": {"kg": "direct"}})
        assert (limits["familyDownloadsPerDay"], limits["familyPlaysPerDay"]) == (150, 300)
        assert {row["platform"]: (row["transport"], row["origin"]) for row in limits["transports"]}["kg"] == (
            "direct", "manual",
        )
        assert call("POST", "/api/source/limits", {"familyPlaysPerDay": 10001}, 400) == {"error": "bad_request"}
        assert call("GET", "/api/source/limits")["familyDownloadsPerDay"] == 150

        assert call("POST", f"/api/sources/{source_id}", {"enabled": False})["enabled"] is False
        assert source_id not in runner.loaded()
        assert call("POST", f"/api/sources/{source_id}/test", expected=409) == {"error": "disabled"}
        assert call("DELETE", f"/api/sources/{source_id}") == {"ok": True}
        assert call("GET", "/api/sources")["sources"] == []
    assert not (tmp_path / "sources" / f"{source_id}.js").exists()
    assert json.loads((tmp_path / "source-limits.json").read_text(encoding="utf-8")) == {
        "familyDownloadsPerDay": 150, "familyPlaysPerDay": 300,
    }
    assert not any(b"do-not-show" in data for data in answers)


def test_source_transport_reports_move_direct_to_relay_for_a_week(tmp_path):
    clock = _Clock()
    runner, store, resolver = _stubbed(tmp_path, clock)

    def modes():
        return {row["platform"]: (row["transport"], row["origin"], row["until"]) for row in resolver.limits()["transports"]}

    assert (modes()["kw"], modes()["kg"]) == (("direct", "default", None), ("relay", "default", None))
    assert resolver.report_transport({"platform": "kw", "transport": "direct", "ok": True}) == {
        "platform": "kw", "transport": "direct",
    }
    assert resolver.report_transport({"platform": "kw", "transport": "direct", "ok": False}) == {
        "platform": "kw", "transport": "relay",
    }
    assert modes()["kw"] == ("relay", "reported", int(clock()) + REPORTED_SECONDS)
    assert Transports(tmp_path, clock).mode("kw") == "relay"
    clock.advance(REPORTED_SECONDS)
    assert modes()["kw"] == ("direct", "default", None)
    for body in (
        {"platform": "xx", "transport": "direct", "ok": False},
        {"platform": "kw", "transport": "direct"},
        {"platform": "kw", "transport": "push", "ok": False},
    ):
        with pytest.raises(ResolveError) as error:
            resolver.report_transport(body)
        assert error.value.code == "bad_request"


def test_source_download_hands_the_address_to_the_queue_and_never_to_the_phone(tmp_path, monkeypatch):
    monkeypatch.setattr(_Handler, "_is_trusted", lambda self: True)
    runner, store, resolver = _stubbed(tmp_path)
    _add(store, runner, "A")
    downloads = _Downloads()
    fields = {"filename": "海阔天空.flac", "directory": "下载", "lyrics": True, "cover": False, "minDurationMs": 60000}

    with _container(tmp_path, store, resolver, downloads=downloads) as port:
        status, _, data = _call(port, "POST", "/api/source/download", {**_song("kw", "1"), **fields, "userAgent": "phone"})
    assert (status, json.loads(data)) == (200, {"id": "job-1", "quality": "320k", "sourceName": "A"})
    assert downloads.jobs == [{"url": "http://127.0.0.1:9/kw/1/320k.mp3", **fields}]

    # A job the queue refuses gives the account its download back.
    with pytest.raises(DownloadError):
        resolver.download(_song("kw", "2"), OWNER, _Downloads(DownloadError("bad_request")))
    assert _today(resolver)[1] == {OWNER: (1, 0)}
    with pytest.raises(ResolveError) as error:
        resolver.download(_song("kw", "3"), OWNER, None)
    assert (error.value.code, error.value.status) == ("download_unconfigured", 503)


@pytest.mark.parametrize(("change", "code"), [
    (lambda body: body.update(platform="xx"), "bad_request"),
    (lambda body: body.update(quality="999k"), "bad_request"),
    (lambda body: body.update(exact="yes"), "bad_request"),
    (lambda body: body.update(purpose="stream"), "bad_request"),
    (lambda body: body.update(musicInfo=[]), "bad_request"),
    (lambda body: body["musicInfo"].pop("songmid"), "bad_request"),
    (lambda body: body.update(platform="tx"), "bad_request"),
    (lambda body: body["musicInfo"].update(padding="x" * 70000), "too_large"),
])
def test_source_resolve_checks_the_song_shape_only(tmp_path, change, code):
    runner, store, resolver = _stubbed(tmp_path)
    _add(store, runner, "A", platforms=("kw", "tx", "kg"))
    body = _song("kw", "1")
    change(body)
    with pytest.raises(ResolveError) as error:
        resolver.resolve(body, OWNER)
    assert error.value.code == code
    assert runner.calls == []
    # iOS sends kg's hash only when it has one.
    song = _song("kg", "2")
    del song["musicInfo"]["hash"]
    assert resolver.resolve(song, OWNER)["transport"] == "relay"
