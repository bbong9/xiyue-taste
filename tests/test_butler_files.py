"""Tests for local-disk ButlerFileOps & /api/butler/files/* endpoints."""

import contextlib
import ipaddress
import json
import threading
import urllib.error
import urllib.request

from analyzer.access import AccessSettings
from analyzer.accounts import AccountStore
from analyzer.butler_files import ButlerFileOps
from analyzer.panel import OutputIndex, PanelState
from analyzer.personal import PersonalStore
from analyzer.serve import make_server

OWNER_TOKEN = "test-owner-files-token"
PASSWORD = "Family-files-password"


def _http(port, method, path, body=None, headers=None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req_headers = {"Content-Type": "application/json"} if data is not None else {}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


@contextlib.contextmanager
def _server(tmp_path, files, accounts=None, personal=None):
    access = AccessSettings(
        None, token=OWNER_TOKEN, network=ipaddress.ip_network("192.168.50.0/24"), host_ip="192.168.50.2",
    )
    server = make_server(
        tmp_path, 0, data=tmp_path, state=PanelState(), index=OutputIndex(tmp_path),
        access=access, accounts=accounts, personal=personal, butler_files=files,
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


def test_butler_files_trash_restore_purge_and_safety(tmp_path):
    library = tmp_path / "downloads"
    album = library / "周杰伦" / "叶惠美"
    album.mkdir(parents=True)
    mp3 = album / "01 晴天.mp3"
    flac = album / "01 晴天.flac"
    lrc = album / "01 晴天.lrc"
    other = album / "02 东风破.mp3"
    other_lrc = album / "02 东风破.lrc"
    mp3.write_bytes(b"mp3-bytes")
    flac.write_bytes(b"flac-bytes")
    lrc.write_bytes(b"[00:01.00]shared lrc")
    other.write_bytes(b"other-mp3")
    other_lrc.write_bytes(b"[00:01.00]own lrc")

    ops = ButlerFileOps(library)
    assert ops.available() is True

    # Resolve full Fly Music NAS paths automatically by suffix matching.
    resolved = ops.resolve_batch([
        "/vol1/1000/Music/周杰伦/叶惠美/01 晴天.mp3",
        "/vol1/1000/Music/周杰伦/叶惠美/02 东风破.mp3",
    ])
    assert resolved["matched"]["/vol1/1000/Music/周杰伦/叶惠美/01 晴天.mp3"] == "周杰伦/叶惠美/01 晴天.mp3"
    assert resolved["prefix"] == "/vol1/1000/Music/"

    # Trash refuses deleting the same file as keepPath.
    same = ops.trash({
        "items": [{"path": "/vol1/1000/Music/周杰伦/叶惠美/01 晴天.mp3", "keepPath": "/vol1/1000/Music/周杰伦/叶惠美/01 晴天.mp3"}],
        "stamp": 1760000000,
    })
    assert same["done"] == 0
    assert same["results"][0]["error"] == "same_file"
    assert mp3.is_file()

    # Trash duplicate mp3 (keeping flac): shared 01 晴天.lrc MUST stay for flac!
    # Also trash 02 东风破.mp3 (no keepPath, unique stem): its own .lrc moves too!
    trashed = ops.trash({
        "items": [
            {"path": "/vol1/1000/Music/周杰伦/叶惠美/01 晴天.mp3", "keepPath": "/vol1/1000/Music/周杰伦/叶惠美/01 晴天.flac"},
            {"path": "/vol1/1000/Music/周杰伦/叶惠美/02 东风破.mp3"},
        ],
        "stamp": 1760000000,
    })
    assert trashed["done"] == 2
    assert trashed["failed"] == 0
    assert not mp3.exists()
    assert flac.is_file()
    assert lrc.is_file(), "共享 .lrc 必须留给保留的 flac"
    assert trashed["results"][0]["lyricsTrash"] is None
    assert not other.exists()
    assert not other_lrc.exists()
    assert trashed["results"][1]["lyricsTrash"] == "周杰伦/叶惠美/.trash/1760000000-02 东风破.lrc"

    # Status & list_trash report the 3 trashed files.
    st = ops.status()
    assert st["trashCount"] == 3
    assert len(ops.list_trash()["items"]) == 3

    # Restore 02 东风破.mp3 and its .lrc back to original paths.
    restored = ops.restore({
        "items": [
            {"trashPath": trashed["results"][1]["relativeTrash"], "restorePath": trashed["results"][1]["relativeFrom"]},
            {"trashPath": trashed["results"][1]["lyricsTrash"], "restorePath": trashed["results"][1]["lyricsFrom"]},
        ],
    })
    assert restored["done"] == 2
    assert other.is_file()
    assert other_lrc.is_file()

    # Security invariant: purge refuses any path outside .trash!
    refused = ops.purge({"paths": ["周杰伦/叶惠美/01 晴天.flac", "../escape.mp3"]})
    assert refused["done"] == 0
    assert refused["failed"] == 2
    assert all(r["error"] == "outside" for r in refused["results"])
    assert flac.is_file()

    # Purge trashed mp3 succeeds, cleans empty .trash folder, and is idempotent on repeat.
    trash_mp3 = trashed["results"][0]["relativeTrash"]
    purged = ops.purge({"paths": [trash_mp3]})
    assert purged["done"] == 1
    assert not (album / ".trash").exists()
    purged_again = ops.purge({"paths": [trash_mp3]})
    assert purged_again["done"] == 1
    assert purged_again["results"][0]["alreadyGone"] is True


def test_butler_files_http_owner_only(tmp_path):
    library = tmp_path / "downloads"
    library.mkdir()
    (library / "a.mp3").write_bytes(b"hello")
    accounts = AccountStore(tmp_path)
    personal = PersonalStore(tmp_path)
    accounts.create("家人", PASSWORD)
    family_token = accounts.login("家人", PASSWORD, "iPhone")["token"]

    owner_headers = {"Authorization": f"Bearer {OWNER_TOKEN}"}
    family_headers = {"Authorization": f"Bearer {family_token}"}
    ops = ButlerFileOps(library)

    with _server(tmp_path, ops, accounts=accounts, personal=personal) as port:
        # Family forbidden on all file mutation & status endpoints.
        status, body = _http(port, "GET", "/api/butler/files/status", headers=family_headers)
        assert status == 403 and body["error"] == "owner_only"
        status, _ = _http(port, "POST", "/api/butler/files/trash", {"items": [{"path": "a.mp3"}]}, headers=family_headers)
        assert status == 403

        # Owner succeeds end-to-end.
        status, body = _http(port, "GET", "/api/butler/files/status", headers=owner_headers)
        assert status == 200 and body["available"] is True
        status, body = _http(port, "POST", "/api/butler/files/trash", {"items": [{"path": "a.mp3"}], "stamp": 100}, headers=owner_headers)
        assert status == 200 and body["done"] == 1
        rel_trash = body["results"][0]["relativeTrash"]
        status, body = _http(port, "POST", "/api/butler/files/purge", {"paths": [rel_trash]}, headers=owner_headers)
        assert status == 200 and body["done"] == 1
