"""Persistent ButlerVaultStore & HTTP endpoints for artist photos, bios, covers, lyrics, and avatars."""

import base64
import contextlib
import ipaddress
import json
import stat
import threading
import urllib.error
import urllib.request

import pytest

from analyzer.access import AccessSettings
from analyzer.accounts import AccountStore
from analyzer.panel import OutputIndex, PanelState
from analyzer.personal import PersonalStore
from analyzer.serve import make_server
from analyzer.vault import ButlerVaultError, ButlerVaultStore, MAX_IMAGE_BYTES

OWNER_TOKEN = "test-owner-vault-token"
PASSWORD = "Family-vault-password"
SAMPLE_JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"
SAMPLE_JPEG_2 = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x02\x00\x02\x00\x00\xff\xd9"
SAMPLE_PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"


class _Clock:
    def __init__(self, now=1_760_000_000):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _http(port, method, path, body=None, headers=None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req_headers = {"Content-Type": "application/json"} if data is not None else {}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, dict(resp.headers.items()), resp.read()
    except urllib.error.HTTPError as err:
        return err.code, dict(err.headers.items()), err.read()


@contextlib.contextmanager
def _server(tmp_path, vault, accounts=None, personal=None):
    access = AccessSettings(
        None, token=OWNER_TOKEN, network=ipaddress.ip_network("192.168.50.0/24"), host_ip="192.168.50.2",
    )
    server = make_server(
        tmp_path, 0, data=tmp_path, state=PanelState(), index=OutputIndex(tmp_path),
        access=access, accounts=accounts, personal=personal, vault=vault,
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


def test_vault_upsert_reload_overwrite_and_delete(tmp_path):
    clock = _Clock()
    store = ButlerVaultStore(tmp_path, clock=clock)

    empty = store.manifest()
    assert empty["revision"] == 0
    assert empty["unchanged"] is False
    assert store.manifest(since=0)["unchanged"] is True

    result = store.upsert({
        "photos": [{"key": "jay chou", "name": "周杰伦", "sourceName": "QQ 音乐", "data": _b64(SAMPLE_JPEG)}],
        "bios": [{"key": "jay chou", "name": "周杰伦", "text": "华语流行歌手", "sourceName": "QQ 音乐"}],
        "covers": [{
            "albumGUID": "alb-1", "trackGUIDs": ["trk-1", "trk-2"],
            "title": "叶惠美", "artist": "周杰伦", "sourceName": "iTunes", "data": _b64(SAMPLE_PNG),
        }],
        "lyrics": [{
            "trackID": "trk-1", "raw": "[00:01.00]晴天", "translatedRaw": "[00:01.00]Sunny Day",
            "sourceName": "网易云", "onlineTitle": "晴天", "onlineArtists": ["周杰伦"],
        }],
        "avatars": [{"key": "fnos:main", "displayName": "飞牛音乐", "data": _b64(SAMPLE_JPEG)}],
    })
    assert result["ok"] is True
    assert result["revision"] == 1
    assert result["saved"] == {"photos": 1, "bios": 1, "covers": 1, "lyrics": 1, "avatars": 1}

    # Verify private file mode on index.json and reload from disk (container restart).
    index_path = tmp_path / "butler-vault" / "index.json"
    assert stat.S_IMODE(index_path.stat().st_mode) == 0o600
    reloaded = ButlerVaultStore(tmp_path, clock=clock)
    manifest = reloaded.manifest()
    assert manifest["revision"] == 1
    assert manifest["counts"] == {"photos": 1, "bios": 1, "covers": 1, "lyrics": 1, "avatars": 1}

    # Re-uploading identical bytes is idempotent and does not bump revision.
    same = reloaded.upsert({
        "photos": [{"key": "jay chou", "name": "周杰伦", "sourceName": "QQ 音乐", "filledAt": clock.now, "data": _b64(SAMPLE_JPEG)}],
    })
    assert same["revision"] == 1
    assert same["saved"]["photos"] == 0

    # Overwriting with new image replaces blob and bumps revision.
    clock.advance(60)
    updated = reloaded.upsert({
        "photos": [{"key": "jay chou", "name": "周杰伦", "sourceName": "相册", "data": _b64(SAMPLE_JPEG_2)}],
    })
    assert updated["revision"] == 2
    assert updated["saved"]["photos"] == 1
    blob_id = reloaded.manifest()["photos"][0]["blobId"]
    data, content_type, _ = reloaded.read_blob("photos", blob_id)
    assert data == SAMPLE_JPEG_2
    assert content_type == "image/jpeg"

    # Batch read lyrics text.
    batch = reloaded.read_lyrics_batch(["trk-1", "missing-track"])
    assert len(batch["items"]) == 1
    assert batch["items"][0]["raw"] == "[00:01.00]晴天"
    assert batch["items"][0]["translatedRaw"] == "[00:01.00]Sunny Day"

    # Delete removes index entry and unlinks blob file on disk.
    deleted = reloaded.delete({"photos": ["jay chou"], "lyrics": ["trk-1"]})
    assert deleted["revision"] == 3
    assert deleted["removed"]["photos"] == 1
    assert deleted["removed"]["lyrics"] == 1
    assert not (tmp_path / "butler-vault" / "photos" / f"{blob_id}.img").exists()
    with pytest.raises(ButlerVaultError) as exc:
        reloaded.read_blob("photos", blob_id)
    assert exc.value.status == 404


def test_vault_http_auth_etag_and_validation(tmp_path):
    accounts = AccountStore(tmp_path)
    personal = PersonalStore(tmp_path)
    accounts.create("家人", PASSWORD)
    family_token = accounts.login("家人", PASSWORD, "iPhone")["token"]
    vault = ButlerVaultStore(tmp_path)

    owner_headers = {"Authorization": f"Bearer {OWNER_TOKEN}"}
    family_headers = {"Authorization": f"Bearer {family_token}"}

    with _server(tmp_path, vault, accounts=accounts, personal=personal) as port:
        # Unauthenticated rejected
        status, _, _ = _http(port, "GET", "/api/butler/vault")
        assert status == 401

        # Owner writes items
        status, _, raw = _http(port, "POST", "/api/butler/vault/items", {
            "photos": [{"key": "a", "name": "歌手A", "sourceName": "相册", "data": _b64(SAMPLE_JPEG)}],
            "lyrics": [{"trackID": "t1", "raw": "[00:00.00]词", "sourceName": "QQ"}],
        }, headers=owner_headers)
        assert status == 200
        body = json.loads(raw)
        assert body["revision"] == 1

        # Family cannot write or delete
        status, _, raw = _http(port, "POST", "/api/butler/vault/items", {
            "bios": [{"key": "a", "name": "歌手A", "text": "恶意修改"}],
        }, headers=family_headers)
        assert status == 403
        assert json.loads(raw)["error"] == "owner_only"

        status, _, _ = _http(port, "POST", "/api/butler/vault/delete", {"photos": ["a"]}, headers=family_headers)
        assert status == 403

        # Family CAN read manifest, blob (with 304 ETag), and lyrics batch
        status, _, raw = _http(port, "GET", "/api/butler/vault", headers=family_headers)
        assert status == 200
        manifest = json.loads(raw)
        blob_id = manifest["photos"][0]["blobId"]

        status, _, raw = _http(port, "GET", "/api/butler/vault?since=1", headers=family_headers)
        assert status == 200
        assert json.loads(raw)["unchanged"] is True

        status, headers, payload = _http(port, "GET", f"/api/butler/vault/blob/photos/{blob_id}", headers=family_headers)
        assert status == 200
        assert payload == SAMPLE_JPEG
        etag = headers.get("ETag") or headers.get("etag")
        assert etag

        status, _, _ = _http(
            port, "GET", f"/api/butler/vault/blob/photos/{blob_id}",
            headers={**family_headers, "If-None-Match": etag},
        )
        assert status == 304

        status, _, raw = _http(
            port, "POST", "/api/butler/vault/lyrics/batch", {"trackIDs": ["t1"]}, headers=family_headers,
        )
        assert status == 200
        assert json.loads(raw)["items"][0]["raw"] == "[00:00.00]词"

        # Validation rejects non-image payload & path traversal blobId
        status, _, raw = _http(port, "POST", "/api/butler/vault/items", {
            "photos": [{"key": "bad", "name": "坏图", "data": _b64(b"not-an-image-at-all")}],
        }, headers=owner_headers)
        assert status == 400
        assert json.loads(raw)["error"] == "unsupported_image"

        status, _, _ = _http(port, "GET", "/api/butler/vault/blob/photos/../../etc/passwd", headers=owner_headers)
        assert status == 404

        # Oversized image rejected with 413
        too_big = b"\xff\xd8\xff" + b"\x00" * MAX_IMAGE_BYTES
        status, _, _ = _http(port, "POST", "/api/butler/vault/items", {
            "photos": [{"key": "big", "name": "大图", "data": _b64(too_big)}],
        }, headers=owner_headers)
        assert status == 413
