"""Family accounts, opaque personal data, per-song quotas and the HTTP identity boundary."""

import contextlib
import hashlib
import ipaddress
import json
import logging
import stat
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from analyzer.accounts import AccountError, AccountStore
from analyzer.access import AccessSettings
from analyzer.ask import AskError
from analyzer.downloads import DownloadError, Downloads
from analyzer.panel import OutputIndex, PanelState
from analyzer.personal import PersonalError, PersonalStore
from analyzer.resolver import CACHE_SECONDS, OWNER, ResolveError, SourceLedger
from analyzer.serve import _Handler, make_server
from test_sources import _Clock, _Downloads, _add, _call, _song, _stubbed

PASSWORD = "Family-test-password"
OWNER_TOKEN = "test-owner-access-token"


def _family(store, name="家人", device="测试设备"):
    account = store.create(name, PASSWORD)
    login = store.login(name, PASSWORD, device)
    return account["id"], login["token"]


def _signed(token, outside=False):
    return {"Authorization": "Bearer " + token, **({"X-Forwarded-For": "203.0.113.9"} if outside else {})}


def _request(port, method, path, body=None, token=None, status=200, outside=False):
    headers = _signed(token, outside) if token else ({"X-Forwarded-For": "203.0.113.9"} if outside else {})
    got, _, payload = _call(port, method, path, body, headers)
    answer = json.loads(payload)
    assert got == status, (method, path, got, answer)
    return answer


@contextlib.contextmanager
def _server(data, accounts, personal, resolver=None, sources=None, downloads=None, asker=None):
    access = AccessSettings(None, token=OWNER_TOKEN, network=ipaddress.ip_network("192.168.50.0/24"), host_ip="192.168.50.2")
    server = make_server(
        data, 0, data=data, state=PanelState(), index=OutputIndex(data),
        accounts=accounts, personal=personal, access=access, resolver=resolver, sources=sources,
        downloads=downloads, asker=asker,
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


def test_accounts_create_login_lookup_logout_and_private_storage(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="xiyue")
    clock = _Clock()
    store = AccountStore(tmp_path, clock)
    account, token = _family(store)
    assert token.startswith("xyd_")
    assert store.lookup(token) == ("ok", account)
    path = tmp_path / "accounts.json"
    raw = path.read_text(encoding="utf-8")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert PASSWORD not in raw and token not in raw
    saved = json.loads(raw)
    password = saved["accounts"][0]["password"]
    assert {key: password[key] for key in ("n", "r", "p")} == {"n": 16384, "r": 8, "p": 1}
    assert password["hash"] == hashlib.scrypt(
        PASSWORD.encode(), salt=bytes.fromhex(password["salt"]), n=16384, r=8, p=1, dklen=32,
    ).hex()
    assert saved["devices"][0]["tokenHash"] == hashlib.sha256(token.encode()).hexdigest()
    assert AccountStore(tmp_path, clock).lookup(token) == ("ok", account)
    panel = json.dumps(store.panel_rows(), ensure_ascii=False)
    assert all(secret not in panel for secret in (token, PASSWORD, password["hash"], password["salt"]))
    store.logout(token)
    assert store.lookup(token) == ("unknown", None)
    assert AccountStore(tmp_path, clock).lookup(token) == ("unknown", None)
    assert all(secret not in caplog.text for secret in (token, PASSWORD, "家人", "测试设备"))


def test_accounts_lock_after_five_failures_and_reset_after_success(tmp_path):
    clock, store = _Clock(), None
    store = AccountStore(tmp_path, clock)
    store.create("家人", PASSWORD)
    for _ in range(5):
        with pytest.raises(AccountError) as error:
            store.login("家人", "wrong-password")
        assert (error.value.code, error.value.status) == ("bad_login", 401)
    with pytest.raises(AccountError) as error:
        store.login("家人", PASSWORD)
    assert (error.value.code, error.value.status) == ("too_many_attempts", 429)
    clock.advance(900)
    assert store.login("家人", PASSWORD)["account"]["name"] == "家人"
    saved = json.loads((tmp_path / "accounts.json").read_text())["accounts"][0]
    assert saved["failures"] == 0 and saved["lockedUntil"] is None
    with pytest.raises(AccountError) as error:
        store.login("家人", "wrong-password")
    assert error.value.status == 401


def test_accounts_global_throttle_after_31_failures(tmp_path):
    clock = _Clock()
    store = AccountStore(tmp_path, clock)
    store.create("家人", PASSWORD)
    for number in range(31):
        with pytest.raises(AccountError) as error:
            store.login(f"不存在{number}", PASSWORD)
        assert error.value.status == 401
    with pytest.raises(AccountError) as error:
        store.login("家人", PASSWORD)
    assert (error.value.code, error.value.status) == ("too_many_attempts", 429)
    clock.advance(600)
    assert store.login("家人", PASSWORD)["token"].startswith("xyd_")


def test_accounts_unknown_name_also_runs_scrypt(tmp_path, monkeypatch):
    store = AccountStore(tmp_path)
    calls = []
    original = hashlib.scrypt

    def scrypt(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(hashlib, "scrypt", scrypt)
    with pytest.raises(AccountError, match="bad_login"):
        store.login("不存在", PASSWORD)
    assert len(calls) == 1
    assert {key: calls[0][key] for key in ("n", "r", "p", "dklen")} == {"n": 16384, "r": 8, "p": 1, "dklen": 32}


def test_accounts_disabled_then_enabled_and_password_kicks_devices(tmp_path, monkeypatch):
    monkeypatch.setattr(_Handler, "_is_trusted", lambda self: True)
    accounts, personal = AccountStore(tmp_path), PersonalStore(tmp_path)
    account, token = _family(accounts)
    second = accounts.login("家人", PASSWORD)["token"]
    with _server(tmp_path, accounts, personal) as port:
        _request(port, "POST", f"/api/accounts/{account}", {"enabled": False})
        assert _request(port, "POST", "/api/account/login", {"name": "家人", "password": PASSWORD}, status=403) == {
            "error": "account_disabled",
        }
        assert _request(port, "GET", "/api/stats", token=token, status=403) == {"error": "account_disabled"}
        assert len(accounts.panel_rows()[0]["devices"]) == 2
        _request(port, "POST", f"/api/accounts/{account}", {"enabled": True})
        _request(port, "GET", "/api/stats", token=token)
        _request(port, "POST", f"/api/accounts/{account}", {"password": "Another-family-password"})
        for old in (token, second):
            assert _request(port, "GET", "/api/stats", token=old, status=401) == {"error": "unauthorized"}
        assert accounts.panel_rows()[0]["devices"] == []


def test_accounts_evict_least_recent_device_and_throttle_last_seen_writes(tmp_path):
    clock = _Clock()
    store = AccountStore(tmp_path, clock)
    account = store.create("家人", PASSWORD)["id"]
    tokens = []
    for number in range(10):
        tokens.append(store.login("家人", PASSWORD, f"设备{number}")["token"])
        clock.advance(1)
    path = tmp_path / "accounts.json"
    before = path.read_bytes()
    assert store.lookup(tokens[0]) == ("ok", account)
    assert path.read_bytes() == before
    clock.advance(600)
    store.lookup(tokens[0])
    assert path.read_bytes() != before
    newest = store.login("家人", PASSWORD, "\n")["token"]
    assert store.lookup(tokens[1]) == ("unknown", None)
    assert store.lookup(tokens[0]) == ("ok", account)
    assert store.lookup(newest) == ("ok", account)
    devices = store.panel_rows()[0]["devices"]
    assert len(devices) == 10 and devices[-1]["name"] == "未命名设备"


def test_accounts_delete_clears_personal_directory_devices_and_indexes(tmp_path, monkeypatch):
    monkeypatch.setattr(_Handler, "_is_trusted", lambda self: True)
    accounts, personal = AccountStore(tmp_path), PersonalStore(tmp_path)
    account, token = _family(accounts)
    personal.append(account, [{"id": "one", "event": {"opaque": "保留原样"}}])
    personal.save_collections(account, {"baseRevision": 0, "document": {"opaque": [1, 2]}})
    with _server(tmp_path, accounts, personal) as port:
        _request(port, "DELETE", f"/api/accounts/{account}")
        assert _request(port, "GET", "/api/stats", token=token, status=401) == {"error": "unauthorized"}
    assert not (tmp_path / "accounts" / account).exists()
    assert personal.listening_status(account) == {"total": 0, "lastSeq": 0}
    assert personal.collections(account)["revision"] == 0
    assert accounts.name_of(account) == "已删除的账户"


def test_accounts_identity_owner_boundaries_and_account_usage(tmp_path, monkeypatch):
    accounts, personal = AccountStore(tmp_path), PersonalStore(tmp_path)
    account, token = _family(accounts)
    runner, sources, resolver = _stubbed(tmp_path)
    _add(sources, runner, "A")
    with _server(tmp_path, accounts, personal, resolver, sources) as port:
        # The real trust check rejects a forwarded request even from a local test server.
        _request(port, "GET", "/api/account/me", outside=True, status=401)
        owner = _request(port, "GET", "/api/account/me", token=OWNER_TOKEN, outside=True)
        assert owner["account"] == {"id": OWNER, "name": "主账户", "owner": True}
        assert owner["limits"] is None
        assert owner["today"] == {"download": 0, "play": 0, "ask": 0}
        assert _request(port, "GET", "/api/accounts", token=OWNER_TOKEN, outside=True, status=403) == {"error": "home_only"}
        _request(port, "POST", "/api/source/resolve", _song("kw", "1"), token=token, outside=True)
        me = _request(port, "GET", "/api/account/me", token=token, outside=True)
        assert me["account"] == {"id": account, "name": "家人", "owner": False}
        assert me["limits"] == {"downloadsPerDay": 200, "playsPerDay": 300}
        assert me["today"] == {"download": 0, "play": 1, "ask": 0}

        monkeypatch.setattr(_Handler, "_is_trusted", lambda self: True)
        assert _request(port, "GET", "/api/account/me")["account"]["id"] == OWNER
        assert _request(port, "GET", "/api/account/me", token="old-owner-token")["account"]["id"] == OWNER
        assert _request(port, "GET", "/api/stats", token="xyd_unknown", status=401) == {"error": "unauthorized"}
        for method, path in (
            ("GET", "/api/sources"), ("GET", "/api/accounts"),
            ("GET", "/api/logs"), ("GET", "/api/logs/download"), ("GET", "/api/llm"),
            ("POST", "/api/llm"), ("POST", "/api/llm/test"), ("POST", "/api/scan"),
            ("POST", "/api/butler/artists"), ("POST", "/api/butler/songs"),
            ("POST", "/api/access"), ("POST", "/api/sources"), ("POST", "/api/source/limits"),
        ):
            assert _request(port, method, path, {}, token=token, status=403) == {"error": "owner_only"}
        usage = _request(port, "GET", "/api/source/usage")
        assert usage["days"][0]["accounts"] == [{"account": account, "download": 0, "play": 1, "ask": 0, "name": "家人"}]
        for path in ("/api/status", "/api/stats", "/api/tracks", "/api/failures", "/api/access", "/api/ask/status", "/api/source/capabilities"):
            _request(port, "GET", path, token=token)


def test_accounts_unconfigured_and_cached_identity_per_request(tmp_path, monkeypatch):
    monkeypatch.setattr(_Handler, "_is_trusted", lambda self: True)
    with _server(tmp_path, None, None) as port:
        for method, path in (("GET", "/api/account/me"), ("POST", "/api/account/login"), ("GET", "/api/accounts")):
            assert _request(port, method, path, {}, status=503) == {"error": "accounts_unconfigured"}
        _request(port, "GET", "/api/stats", token="xyd_unknown", status=401)
        _request(port, "GET", "/api/stats")
    accounts, personal = AccountStore(tmp_path), PersonalStore(tmp_path)
    _, token = _family(accounts)
    calls = []
    lookup = accounts.lookup

    def count_lookup(token):
        calls.append(token)
        return lookup(token)

    monkeypatch.setattr(accounts, "lookup", count_lookup)
    with _server(tmp_path, accounts, personal) as port:
        _request(port, "GET", "/api/account/me", token=token)
        assert len(calls) == 1
        _request(port, "GET", "/api/account/me", token=token)
        assert len(calls) == 2


def test_accounts_login_logout_revoke_and_panel_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(_Handler, "_is_trusted", lambda self: True)
    accounts, personal = AccountStore(tmp_path), PersonalStore(tmp_path)
    with _server(tmp_path, accounts, personal) as port:
        account = _request(port, "POST", "/api/accounts", {"name": " 家人 ", "password": PASSWORD})["id"]
        login = _request(port, "POST", "/api/account/login", {"name": "家人", "password": PASSWORD, "device": "一号手机"}, outside=True)
        token = login["token"]
        assert login["account"] == {"id": account, "name": "家人"}
        _request(port, "POST", "/api/me/listening", {"events": [{"id": "one", "event": {"data": 7}}]}, token)
        _request(port, "POST", "/api/me/collections", {
            "baseRevision": 0, "document": {"data": "不解读"},
            "summary": {"liked": 3, "playlists": 2, "tracks": 5},
        }, token)
        rows = _request(port, "GET", "/api/accounts")["accounts"]
        assert rows[0]["owner"] is True and rows[0]["devices"] == []
        family = rows[1]
        assert family["listening"] == {"total": 1, "lastSeq": 1}
        assert family["collections"]["summary"] == {"liked": 3, "playlists": 2, "tracks": 5}
        assert family["lastSyncAt"] is not None and family["devices"][0]["name"] == "一号手机"
        assert "tokenHash" not in json.dumps(rows) and "password" not in json.dumps(rows)
        device = family["devices"][0]["id"]
        _request(port, "DELETE", f"/api/accounts/{account}/devices/{device}")
        _request(port, "GET", "/api/account/me", token=token, status=401)
        token = _request(port, "POST", "/api/account/login", {"name": "家人", "password": PASSWORD})["token"]
        _request(port, "POST", "/api/account/logout", token=token)
        _request(port, "GET", "/api/account/me", token=token, status=401)
        assert _request(port, "POST", "/api/account/logout", status=400) == {"error": "not_a_device"}
        for path in ("/api/accounts/../../", "/api/accounts/owner", "/api/accounts/" + "a" * 17):
            _request(port, "DELETE", path, status=404)


def test_personal_listening_deduplicates_pages_and_preserves_opaque_events(tmp_path):
    personal = PersonalStore(tmp_path)
    events = [{"id": f"e{number}", "event": {"anything": [number, {"中文": True}], "noSchema": None}} for number in range(7)]
    assert personal.append(OWNER, events + [events[0]]) == {"accepted": 7, "duplicates": 1, "total": 7, "lastSeq": 7}
    assert personal.append(OWNER, events)["duplicates"] == 7
    after, found = 0, []
    while True:
        page = personal.listening(OWNER, after, 2)
        assert len(page["events"]) <= 2
        found.extend(page["events"])
        after = page["lastSeq"]
        if not page["hasMore"]:
            break
    assert found == [{"seq": number + 1, **event} for number, event in enumerate(events)]
    assert personal.listening(OWNER, after)["events"] == []
    assert len(personal.listening(OWNER, limit=0)["events"]) == 1
    for path in (tmp_path / "accounts", tmp_path / "accounts" / OWNER):
        assert stat.S_IMODE(path.stat().st_mode) == 0o700
    assert stat.S_IMODE((tmp_path / "accounts" / OWNER / "listening.jsonl").stat().st_mode) == 0o600
    assert PersonalStore(tmp_path).append(OWNER, events)["duplicates"] == 7


@pytest.mark.parametrize("bad", [
    {"id": "bad/id", "event": {}}, {"id": "", "event": {}}, {"id": "x" * 65, "event": {}},
    {"id": "new", "event": []}, {"id": "new", "event": {"text": "中" * 1400}},
])
def test_personal_listening_rejects_whole_batch(tmp_path, bad):
    personal = PersonalStore(tmp_path)
    personal.append(OWNER, [{"id": "existing", "event": {}}])
    with pytest.raises(PersonalError, match="bad_request"):
        personal.append(OWNER, [{"id": "valid", "event": {}}, bad])
    assert personal.listening_status(OWNER) == {"total": 1, "lastSeq": 1}
    assert [event["id"] for event in personal.listening(OWNER)["events"]] == ["existing"]
    with pytest.raises(PersonalError):
        personal.append(OWNER, [{"id": "same", "event": {}}] * 501)


def test_personal_compaction_retains_sequence_and_restarts(tmp_path):
    personal = PersonalStore(tmp_path, max_events=5, kept_events=3)
    personal.append(OWNER, [{"id": f"e{number}", "event": {"n": number}} for number in range(6)])
    assert personal.listening_status(OWNER) == {"total": 3, "lastSeq": 6}
    assert [event["seq"] for event in personal.listening(OWNER)["events"]] == [4, 5, 6]
    reopened = PersonalStore(tmp_path, max_events=5, kept_events=3)
    assert reopened.append(OWNER, [{"id": "e5", "event": {}}, {"id": "new", "event": {}}]) == {
        "accepted": 1, "duplicates": 1, "total": 4, "lastSeq": 7,
    }
    assert reopened.listening(OWNER, after=6)["events"] == [{"seq": 7, "id": "new", "event": {}}]


def test_personal_listening_survives_a_bad_line_and_a_write_cut_short(tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger="xiyue")
    PersonalStore(tmp_path).append(OWNER, [{"id": "a", "event": {"n": 1}}])
    path = tmp_path / "accounts" / OWNER / "listening.jsonl"
    with path.open("ab") as file:
        file.write(b"not json\n" + json.dumps({"seq": 2, "id": "b", "at": 1, "event": {}}).encode() + b"\n")
        file.write(b'{"seq": 3, "id": "c"')

    reopened = PersonalStore(tmp_path)
    assert reopened.listening_status(OWNER) == {"total": 2, "lastSeq": 2}
    assert "PERSONAL-INDEX account=owner skipped=1 cut=是" in caplog.text
    assert path.read_bytes().endswith(b"\n")
    assert reopened.append(OWNER, [{"id": "c", "event": {"n": 3}}])["lastSeq"] == 3
    assert [event["id"] for event in PersonalStore(tmp_path).listening(OWNER)["events"]] == ["a", "b", "c"]


def test_personal_account_isolation_and_concurrent_append_and_revision(tmp_path):
    personal = PersonalStore(tmp_path)
    other = "a" * 16
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: personal.append(OWNER, [{"id": "same", "event": {"opaque": 1}}]), range(10)))
    assert personal.listening_status(OWNER) == {"total": 1, "lastSeq": 1}
    assert personal.listening_status(other) == {"total": 0, "lastSeq": 0}

    def save(_):
        try:
            return personal.save_collections(OWNER, {"baseRevision": 0, "document": {"unknown": [True, None]}})
        except PersonalError as error:
            return error.code

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(save, range(4)))
    assert results.count("conflict") == 3
    assert personal.collections(OWNER)["revision"] == 1
    assert personal.collections(other) == {"revision": 0, "updatedAt": None, "summary": None, "document": None}


def test_personal_http_limits_revisions_summary_and_account_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(_Handler, "_is_trusted", lambda self: True)
    accounts, personal = AccountStore(tmp_path), PersonalStore(tmp_path)
    account, token = _family(accounts)
    _, other = _family(accounts, "另一位")
    document = {"anUnknownField": [{"原样": [1, None, False]}]}
    summary = {"liked": 1, "playlists": 2, "tracks": 3}
    with _server(tmp_path, accounts, personal) as port:
        assert _request(port, "GET", "/api/me/collections", token=token)["revision"] == 0
        saved = _request(port, "POST", "/api/me/collections", {"baseRevision": 0, "document": document, "summary": summary}, token)
        assert saved["revision"] == 1 and type(saved["updatedAt"]) is int
        assert _request(port, "GET", "/api/me/collections", token=token) == {**saved, "summary": summary, "document": document}
        assert _request(port, "POST", "/api/me/collections", {"baseRevision": 0, "document": {}}, token, status=409) == {
            "error": "conflict", **saved,
        }
        for bad in ({}, {"liked": True, "playlists": 0, "tracks": 0}, {"liked": 0, "playlists": -1, "tracks": 0}):
            _request(port, "POST", "/api/me/collections", {"baseRevision": 1, "document": {}, "summary": bad}, token, status=400)
        status, _, data = _call(
            port, "POST", "/api/me/collections", {},
            {**_signed(token), "Content-Length": str(6 * 1024 * 1024 + 1)},
        )
        assert status == 413 and json.loads(data) == {"error": "too_large"}
        assert _request(port, "GET", "/api/me/collections", token=other)["revision"] == 0
        _request(port, "POST", "/api/me/listening", {"events": [{"id": "a", "event": {}}, {"id": "b", "event": {"x": 2}}]}, token)
        page = _request(port, "GET", "/api/me/listening?after=0&limit=1", token=token)
        assert page == {"events": [{"seq": 1, "id": "a", "event": {}}], "lastSeq": 1, "hasMore": True}
        assert _request(port, "GET", "/api/me/listening?after=1&limit=1", token=token)["events"][0]["id"] == "b"
        assert _request(port, "GET", "/api/me/listening", token=other)["events"] == []
        _request(port, "POST", "/api/me/listening", {"events": [{"id": "c", "event": {}}, {"id": "/", "event": {}}]}, token, status=400)
        assert personal.listening_status(account)["total"] == 2
        status, _, data = _call(
            port, "POST", "/api/me/listening", {},
            {**_signed(token), "Content-Length": str(2 * 1024 * 1024 + 1)},
        )
        assert status == 413 and json.loads(data) == {"error": "too_large"}
        status, _, data = _call(
            port, "POST", "/api/account/login", {}, {"Content-Length": str(4096 + 1)},
        )
        assert status == 413 and json.loads(data) == {"error": "too_large"}
    assert stat.S_IMODE((tmp_path / "accounts" / account / "collections.json").stat().st_mode) == 0o600


def test_downloads_filter_cancel_and_both_submission_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(_Handler, "_is_trusted", lambda self: True)
    # Keep the real queue idle; no download or address lookup leaves the test process.
    monkeypatch.setattr(Downloads, "_work", lambda self: None)
    accounts, personal = AccountStore(tmp_path), PersonalStore(tmp_path)
    account, token = _family(accounts)
    other_account, other = _family(accounts, "另一位")
    downloads = Downloads(tmp_path)
    runner, sources, resolver = _stubbed(tmp_path)
    _add(sources, runner, "A")
    body = {"url": "https://example.invalid/song.mp3", "filename": "song.mp3"}
    with _server(tmp_path, accounts, personal, resolver, sources, downloads) as port:
        owner_job = _request(port, "POST", "/api/downloads", body)["id"]
        family_job = _request(port, "POST", "/api/downloads", body, token)["id"]
        other_job = _request(port, "POST", "/api/source/download", {**_song("kw", "1"), "filename": "song.mp3"}, other)["id"]
        jobs = _request(port, "GET", "/api/downloads", token=token)["jobs"]
        assert [(job["id"], job["account"]) for job in jobs] == [(family_job, account)]
        jobs = _request(port, "GET", "/api/downloads")["jobs"]
        assert {job["id"]: (job["account"], job["accountName"]) for job in jobs} == {
            owner_job: (OWNER, ""), family_job: (account, "家人"), other_job: (other_account, "另一位"),
        }
        for denied in (owner_job, other_job):
            assert _request(port, "POST", "/api/downloads/cancel", {"id": denied}, token) == {"ok": False}
        assert _request(port, "POST", "/api/downloads/cancel", {"id": family_job}, token) == {"ok": True}
        assert _request(port, "POST", "/api/downloads/cancel", {"id": other_job}) == {"ok": True}
        assert next(job for job in downloads.snapshot() if job["id"] == owner_job)["state"] == "queued"


def test_ledger_counts_unique_songs_per_day_and_refunds(tmp_path):
    clock = _Clock()
    ledger = SourceLedger(tmp_path, clock)
    ledger.set_limits({"familyPlaysPerDay": 1, "familyDownloadsPerDay": 1})
    receipt = ledger.charge("family", "play", "kw|one")
    assert receipt is not None
    assert ledger.charge("family", "play", "kw|one") is None
    ledger.charge("family", "download", "kw|one")
    assert ledger.charge("family", "download", "kw|one") is None
    with pytest.raises(ResolveError, match="quota_exceeded"):
        ledger.charge("family", "play", "kw|two")
    ledger.refund(receipt)
    saved = json.loads((tmp_path / "source-usage.json").read_text())
    assert saved["days"][ledger.today()]["songs"]["family"]["play"] == []
    ledger.charge("family", "play", "kw|one")
    for _ in range(3):
        ledger.note("family", "ask")
    assert ledger.recent(1)[0]["accounts"]["family"] == {"download": 1, "play": 1, "ask": 3}
    reopened = SourceLedger(tmp_path, clock)
    assert reopened.charge("family", "play", "kw|one") is None
    before = reopened.today()
    clock.advance(12 * 3600)  # Noon to the next Shanghai midnight.
    reopened.charge("family", "play", "kw|one")
    saved = json.loads((tmp_path / "source-usage.json").read_text())["days"]
    assert "songs" not in saved[before]
    assert saved[before]["accounts"]["family"]["play"] == 1
    assert saved[reopened.today()]["songs"]["family"]["play"] == [hashlib.sha256(b"kw|one").hexdigest()[:16]]
    assert "kw|one" not in json.dumps(saved)


def test_resolver_replays_at_limit_after_cache_expiry_and_refunds_failed_song(tmp_path):
    clock = _Clock()
    runner, sources, resolver = _stubbed(tmp_path, clock)
    source = _add(sources, runner, "A")
    resolver.set_limits({"familyPlaysPerDay": 1})
    resolver.resolve(_song("kw", "1"), "family")
    resolver.resolve(_song("kw", "1", "128k"), "family")
    clock.advance(CACHE_SECONDS)
    resolver.resolve(_song("kw", "1"), "family")
    assert resolver.ledger.recent(1)[0]["accounts"]["family"]["play"] == 1
    with pytest.raises(ResolveError, match="quota_exceeded"):
        resolver.resolve(_song("kw", "2"), "family")
    sources.update(source, {"enabled": False})
    with pytest.raises(ResolveError):
        resolver.resolve(_song("kw", "new"), OWNER)
    assert resolver.ledger.recent(1)[0]["accounts"][OWNER]["play"] == 0
    sources.update(source, {"enabled": True})
    resolver.resolve(_song("kw", "new"), OWNER)
    assert resolver.ledger.recent(1)[0]["accounts"][OWNER]["play"] == 1
    downloads = _Downloads()
    resolver.download(_song("kw", "new"), OWNER, downloads)
    assert downloads.account == OWNER
    # Refusing a repeated download has no receipt and must not refund its earlier successful count.
    with pytest.raises(DownloadError):
        resolver.download(_song("kw", "new"), OWNER, _Downloads(DownloadError("bad_request")))
    assert resolver.ledger.recent(1)[0]["accounts"][OWNER]["download"] == 1


def test_accounts_ask_counts_success_only_without_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(_Handler, "_is_trusted", lambda self: True)
    accounts, personal = AccountStore(tmp_path), PersonalStore(tmp_path)
    account, token = _family(accounts)
    _, sources, resolver = _stubbed(tmp_path)
    resolver.set_limits({"familyPlaysPerDay": 0, "familyDownloadsPerDay": 0})

    class Asker:
        def ask(self, query, taste, part):
            if query == "失败":
                raise AskError("ask_failed", 502)
            return {"picks": [], "reason": "空"}

    with _server(tmp_path, accounts, personal, resolver, sources, asker=Asker()) as port:
        for _ in range(3):
            _request(port, "POST", "/api/ask", {"q": "找歌"}, token)
        _request(port, "POST", "/api/ask", {"q": "失败"}, token, status=502)
        assert _request(port, "GET", "/api/account/me", token=token)["today"] == {"download": 0, "play": 0, "ask": 3}
    with _server(tmp_path, accounts, personal, asker=Asker()) as port:
        _request(port, "POST", "/api/ask", {"q": "找歌"}, token)
    assert resolver.ledger.recent(1)[0]["accounts"][account]["ask"] == 3


def test_accounts_validation_limits_and_bad_index_preserves_original(tmp_path, caplog):
    store = AccountStore(tmp_path)
    for name in ("", "owner", "OWNER", "主账户", "a/b", "a\\b", "a\nb", "x" * 21):
        with pytest.raises(AccountError, match="bad_name"):
            store.create(name, PASSWORD)
    for password in ("short", "x" * 129, "long\npassword"):
        with pytest.raises(AccountError, match="bad_password"):
            store.create("家人", password)
    store.create("Family", PASSWORD)
    with pytest.raises(AccountError, match="name_taken"):
        store.create(" family ", PASSWORD)
    for number in range(19):
        store.create(f"家人{number}", PASSWORD)
    with pytest.raises(AccountError) as error:
        store.create("超额", PASSWORD)
    assert (error.value.code, error.value.status) == ("too_many_accounts", 409)
    path = tmp_path / "accounts.json"
    for raw in (b"not-json", b'{"accounts": [{}], "devices": []}'):
        path.write_bytes(raw)
        reopened = AccountStore(tmp_path)
        assert reopened.panel_rows() == []
        assert reopened.lookup("xyd_unknown") == ("unknown", None)
        with pytest.raises(AccountError):
            reopened.login("Family", PASSWORD)
        assert path.read_bytes() == raw
    assert "ACCOUNT-INDEX unreadable" in caplog.text
    reopened.create("新家人", PASSWORD)
    assert json.loads(path.read_text())["accounts"][0]["name"] == "新家人"
