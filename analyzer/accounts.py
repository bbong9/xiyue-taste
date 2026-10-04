"""Owner login names, family accounts and hashed device credentials; access settings stay separate."""

import collections
import hashlib
import hmac
import json
import re
import secrets
import threading
import time
from pathlib import Path

from .log import LOGGER
from .sources import write_private

ACCOUNT_ID = re.compile(r"[0-9a-f]{16}")
_HEX32 = re.compile(r"[0-9a-f]{32}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_PARAMETERS = {"n": 16384, "r": 8, "p": 1}


class AccountError(Exception):
    def __init__(self, code, status=400):
        super().__init__(code)
        self.code, self.status = code, status


def _name(value):
    if not isinstance(value, str):
        raise AccountError("bad_name")
    value = value.strip()
    if (
        not 1 <= len(value) <= 20 or value.casefold() in ("owner", "主账户")
        or any(c in "/\\" or ord(c) < 32 or 127 <= ord(c) <= 159 for c in value)
    ):
        raise AccountError("bad_name")
    return value


def _password(value):
    if not isinstance(value, str) or not 8 <= len(value) <= 128 or not value.isprintable():
        raise AccountError("bad_password")
    return value


def _digest(password, salt):
    return hashlib.scrypt(password.encode("utf-8"), salt=bytes.fromhex(salt), dklen=32, **_PARAMETERS).hex()


def _password_record(password):
    salt = secrets.token_hex(16)
    return {"salt": salt, **_PARAMETERS, "hash": _digest(password, salt)}


_DUMMY_SALT = "00" * 16
_DUMMY = {"salt": _DUMMY_SALT, "hash": _digest("unused-login-password", _DUMMY_SALT)}


def _valid_index(saved):
    if not isinstance(saved, dict) or not isinstance(saved.get("accounts"), list) or not isinstance(saved.get("devices"), list):
        return False
    ids, names, devices, tokens = set(), set(), set(), set()
    if len(saved["accounts"]) > 21:
        return False
    try:
        for row in saved["accounts"]:
            password = row["password"]
            if (
                (row["id"] != "owner" and not ACCOUNT_ID.fullmatch(row["id"])) or row["id"] in ids
                or (row["id"] == "owner" and row["enabled"] is not True)
                or _name(row["name"]) != row["name"] or row["name"].casefold() in names
                or type(row["enabled"]) is not bool or type(row["createdAt"]) is not int
                or type(row["failures"]) is not int or row["failures"] < 0
                or (row["lockedUntil"] is not None and type(row["lockedUntil"]) is not int)
                or not _HEX32.fullmatch(password["salt"]) or not _HEX64.fullmatch(password["hash"])
                or any(type(password[key]) is not int or password[key] != value for key, value in _PARAMETERS.items())
            ):
                return False
            ids.add(row["id"])
            names.add(row["name"].casefold())
        if len(ids - {"owner"}) > 20:
            return False
        counts = collections.Counter()
        for row in saved["devices"]:
            if (
                not ACCOUNT_ID.fullmatch(row["id"]) or row["id"] in devices
                or row["account"] not in ids or not _HEX64.fullmatch(row["tokenHash"])
                or row["tokenHash"] in tokens or not isinstance(row["name"], str)
                or not 1 <= len(row["name"]) <= 40 or not row["name"].isprintable()
                or type(row["createdAt"]) is not int or type(row["lastSeenAt"]) is not int
            ):
                return False
            counts[row["account"]] += 1
            devices.add(row["id"])
            tokens.add(row["tokenHash"])
        return all(count <= 10 for count in counts.values())
    except (KeyError, TypeError, AccountError):
        return False


class AccountStore:
    def __init__(self, data, clock=time.time):
        self._path = Path(data) / "accounts.json"
        self._clock = clock
        self._lock = threading.Lock()
        self._failures = collections.deque()
        self._replaced = collections.OrderedDict()
        self._last_save = clock()
        self._accounts, self._devices = [], []
        try:
            saved = json.loads(self._path.read_text(encoding="utf-8"))
            if not _valid_index(saved):
                raise ValueError("invalid index")
        except FileNotFoundError:
            return
        except (OSError, ValueError):
            LOGGER.warning("ACCOUNT-INDEX unreadable")
            return
        self._accounts, self._devices = saved["accounts"], saved["devices"]

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        write_private(self._path, json.dumps({
            "accounts": self._accounts, "devices": self._devices,
        }, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        self._last_save = self._clock()

    def _find(self, account_id):
        for row in self._accounts:
            if row["id"] == account_id:
                return row
        raise AccountError("not_found", 404)

    def _device(self, token):
        hashed = hashlib.sha256(token.encode("utf-8")).hexdigest()
        return next((row for row in self._devices if hmac.compare_digest(row["tokenHash"], hashed)), None)

    def create(self, name, password):
        name, password = _name(name), _password(password)
        with self._lock:
            if any(row["name"].casefold() == name.casefold() for row in self._accounts):
                raise AccountError("name_taken", 409)
            if sum(row["id"] != "owner" for row in self._accounts) >= 20:
                raise AccountError("too_many_accounts", 409)
            row = {
                "id": secrets.token_hex(8), "name": name, "enabled": True, "createdAt": int(self._clock()),
                "password": _password_record(password), "failures": 0, "lockedUntil": None,
            }
            self._accounts.append(row)
            self._save()
        LOGGER.info("ACCOUNT-CHANGE action=create account=%s", row["id"])
        return {"id": row["id"], "name": row["name"]}

    def set_owner_login(self, name, password):
        name, password = _name(name), _password(password)
        with self._lock:
            if any(row["id"] != "owner" and row["name"].casefold() == name.casefold() for row in self._accounts):
                raise AccountError("name_taken", 409)
            row = next((row for row in self._accounts if row["id"] == "owner"), None)
            if row is None:
                row = {"id": "owner", "enabled": True, "createdAt": int(self._clock())}
                self._accounts.append(row)
            row.update(name=name, password=_password_record(password), failures=0, lockedUntil=None)
            self._devices = [device for device in self._devices if device["account"] != "owner"]
            self._save()
        LOGGER.info("ACCOUNT-CHANGE action=owner-login account=owner")

    def login(self, name, password, device_name=None):
        with self._lock:
            now = int(self._clock())
            while self._failures and now - self._failures[0] >= 600:
                self._failures.popleft()
            row = next((
                row for row in self._accounts
                if isinstance(name, str) and row["name"].casefold() == name.strip().casefold()
            ), None)
            account_id = row["id"] if row is not None else "-"
            if len(self._failures) > 30:
                reason, code, status = "throttled", "too_many_attempts", 429
            elif row is not None and row["lockedUntil"] is not None and now < row["lockedUntil"]:
                self._failures.append(now)
                reason, code, status = "locked", "too_many_attempts", 429
            else:
                if row is not None and row["lockedUntil"] is not None:
                    row.update(failures=0, lockedUntil=None)
                record = row["password"] if row is not None else _DUMMY
                candidate = password if isinstance(password, str) else ""
                matches = hmac.compare_digest(_digest(candidate, record["salt"]), record["hash"])
                if row is None or not matches:
                    self._failures.append(now)
                    if row is not None:
                        row["failures"] += 1
                        if row["failures"] >= 5:
                            row["lockedUntil"] = now + 900
                        self._save()
                    reason, code, status = "bad_login", "bad_login", 401
                elif not row["enabled"]:
                    self._failures.append(now)
                    reason, code, status = "disabled", "account_disabled", 403
                else:
                    row.update(failures=0, lockedUntil=None)
                    device_name = device_name.strip() if isinstance(device_name, str) else ""
                    if not 1 <= len(device_name) <= 40 or not device_name.isprintable():
                        device_name = "未命名设备"
                    token = "xyd_" + secrets.token_urlsafe(32)
                    device = {
                        "id": secrets.token_hex(8), "account": account_id,
                        "tokenHash": hashlib.sha256(token.encode("utf-8")).hexdigest(), "name": device_name,
                        "createdAt": now, "lastSeenAt": now,
                    }
                    own = [item for item in self._devices if item["account"] == account_id]
                    for old in own:
                        self._replaced[old["tokenHash"]] = account_id
                    while len(self._replaced) > 200:
                        self._replaced.popitem(last=False)
                    self._devices = [item for item in self._devices if item["account"] != account_id]
                    self._devices.append(device)
                    self._save()
                    LOGGER.info("ACCOUNT-LOGIN ok account=%s device=%s replaced=%s", account_id, device["id"], len(own))
                    return {"token": token, "account": {"id": account_id, "name": row["name"]}}
            LOGGER.info("ACCOUNT-LOGIN fail reason=%s account=%s", reason, account_id)
            raise AccountError(code, status)

    def lookup(self, token):
        with self._lock:
            device = self._device(token)
            if device is None:
                account_id = self._replaced.get(hashlib.sha256(token.encode("utf-8")).hexdigest())
                if account_id is not None:
                    return "replaced", account_id
                return "unknown", None
            row = self._find(device["account"])
            device["lastSeenAt"] = int(self._clock())
            if self._clock() - self._last_save >= 600:
                self._save()
            return ("ok" if row["enabled"] else "disabled"), row["id"]

    def logout(self, token):
        with self._lock:
            device = self._device(token)
            if device is None:
                return
            self._devices.remove(device)
            self._save()
        LOGGER.info("ACCOUNT-CHANGE action=logout account=%s", device["account"])

    def set_password(self, account_id, password):
        password = _password(password)
        with self._lock:
            row = self._find(account_id)
            row.update(password=_password_record(password), failures=0, lockedUntil=None)
            self._devices = [device for device in self._devices if device["account"] != account_id]
            self._save()
        LOGGER.info("ACCOUNT-CHANGE action=password account=%s", account_id)

    def set_enabled(self, account_id, enabled):
        if account_id == "owner" or type(enabled) is not bool:
            raise AccountError("bad_request")
        with self._lock:
            self._find(account_id)["enabled"] = enabled
            self._save()
        LOGGER.info("ACCOUNT-CHANGE action=%s account=%s", "enable" if enabled else "disable", account_id)

    def delete(self, account_id):
        if account_id == "owner":
            raise AccountError("bad_request")
        with self._lock:
            self._accounts.remove(self._find(account_id))
            self._devices = [device for device in self._devices if device["account"] != account_id]
            for hashed in [key for key, value in self._replaced.items() if value == account_id]:
                del self._replaced[hashed]
            self._save()
        LOGGER.info("ACCOUNT-CHANGE action=delete account=%s", account_id)

    def revoke_device(self, account_id, device_id):
        with self._lock:
            self._find(account_id)
            device = next((item for item in self._devices if item["id"] == device_id and item["account"] == account_id), None)
            if device is None:
                raise AccountError("not_found", 404)
            self._devices.remove(device)
            self._save()
        LOGGER.info("ACCOUNT-CHANGE action=revoke account=%s", account_id)

    def name_of(self, account_id):
        if account_id == "owner":
            return "主账户"
        with self._lock:
            return next((row["name"] for row in self._accounts if row["id"] == account_id), "已删除的账户")

    def panel_rows(self):
        with self._lock:
            return [{
                **{key: row[key] for key in ("id", "name", "enabled", "createdAt")},
                "devices": [
                    {key: device[key] for key in ("id", "name", "createdAt", "lastSeenAt")}
                    for device in self._devices if device["account"] == row["id"]
                ],
            } for row in self._accounts if row["id"] != "owner"]

    def owner_login_name(self):
        with self._lock:
            return next((row["name"] for row in self._accounts if row["id"] == "owner"), None)

    def owner_panel(self):
        with self._lock:
            return {
                "loginName": next((row["name"] for row in self._accounts if row["id"] == "owner"), None),
                "devices": [
                    {key: device[key] for key in ("id", "name", "createdAt", "lastSeenAt")}
                    for device in self._devices if device["account"] == "owner"
                ],
            }
