import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from .llm import chat_json


class SettingsError(Exception):
    pass


class LLMSettings:
    """The model address, name and key: the environment's, then whatever the
    panel saved over them. The key is written and used, never handed back."""

    def __init__(self, path, api_key, base_url, model, targets=(), urlopen=urllib.request.urlopen):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._targets = tuple(targets)
        self._urlopen = urlopen
        self._api_key, self._base_url, self._model = api_key, base_url, model
        try:
            saved = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved = {}
        if isinstance(saved, dict):
            for name in ("api_key", "base_url", "model"):
                if isinstance(saved.get(name), str) and saved[name]:
                    setattr(self, "_" + name, saved[name])
        self._apply()

    def _apply(self):
        for target in self._targets:
            target.configure(self._api_key, self._base_url, self._model)

    def status(self):
        with self._lock:
            return {"configured": bool(self._api_key), "baseURL": self._base_url, "model": self._model}

    def update(self, base_url, model, api_key=None):
        """Saves the address and model; a key only when one is given."""
        if not isinstance(base_url, str) or not isinstance(model, str):
            raise SettingsError
        base_url, model = base_url.strip().rstrip("/"), model.strip()
        if not base_url.startswith(("https://", "http://")) or len(base_url) > 300 or any(c.isspace() for c in base_url):
            raise SettingsError
        if not 1 <= len(model) <= 100 or any(c.isspace() for c in model):
            raise SettingsError
        if api_key is not None:
            if not isinstance(api_key, str):
                raise SettingsError
            api_key = api_key.strip()
            if len(api_key) > 300 or any(c.isspace() or not c.isprintable() for c in api_key):
                raise SettingsError
        with self._lock:
            self._base_url, self._model = base_url, model
            if api_key:
                self._api_key = api_key
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._path.with_suffix(".tmp")
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                json.dump({"api_key": self._api_key, "base_url": base_url, "model": model}, file)
            os.replace(temporary, self._path)
            self._apply()

    def test(self):
        """One tiny question to the model: whether it answers, and how fast."""
        with self._lock:
            api_key, base_url, model = self._api_key, self._base_url, self._model
        if not api_key:
            return {"ok": False, "error": "unconfigured", "detail": ""}
        began = time.monotonic()
        try:
            chat_json(
                api_key, base_url, model, '只输出这个 JSON，不要任何其他文字：{"ok": true}', "测试",
                20, 30, self._urlopen,
            )
        except urllib.error.HTTPError as error:
            try:
                detail = error.read(400).decode("utf-8", "replace")
            except OSError:
                detail = ""
            return {"ok": False, "error": f"http_{error.code}", "detail": detail.replace(api_key, "***")[:200]}
        except (urllib.error.URLError, OSError):
            return {"ok": False, "error": "unreachable", "detail": ""}
        except (ValueError, KeyError, IndexError, TypeError):
            return {"ok": False, "error": "bad_answer", "detail": ""}
        return {"ok": True, "ms": round((time.monotonic() - began) * 1000), "model": model}
