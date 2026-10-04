"""The Node source runner seen from Python: starts it, restarts it when it dies, and makes the
network requests its scripts ask for.

Each enabled source script runs in its own vm context inside one Node process
(lxrunner/runner.mjs, which describes the line protocol). vm is not a security boundary: Node
runs under its permission model, may read only the runner directory, gets an empty environment
and never touches the network. Every lx.request comes here and goes out through lxnet by the
iOS rules. Script text, request headers and bodies, and full addresses are never logged; a log
line names a host and a status code at most.
"""

import collections
import concurrent.futures
import json
import re
import shutil
import subprocess
import threading
import time
import urllib.parse
import uuid
from pathlib import Path

from . import lxnet
from .log import LOGGER

RUNNER_DIR = Path(__file__).resolve().parent / "lxrunner"
PLATFORMS = ("kw", "kg", "tx", "wy", "mg")
QUALITIES = ("128k", "320k", "flac", "flac24bit", "hires", "atmos", "atmos_plus", "master")
# iOS: 12 s for a script to send init, 35 s for one musicUrl.
LOAD_TIMEOUT = 12.0
CALL_TIMEOUT = 35.0
READY_TIMEOUT = 10.0
# How long past its own deadline a load or call may stay unanswered before Node counts as hung.
GRACE = 3.0
MAX_LINE_BYTES = 2 * 1024 * 1024
MAX_REQUESTS_PER_SOURCE = 4
HEAP_MB = 256
RESTART_DELAYS = (1, 2, 5, 10, 30, 60)
# A run at least this long starts the restart delays over.
STABLE_SECONDS = 60
STDERR_LINES_PER_MINUTE = 20
_CODE = re.compile(r"[a-z_]{1,40}")
_SOURCE_ID = re.compile(r"[0-9a-f]{1,32}")
_UNPRINTABLE = re.compile(r"[^\x20-\x7e]")
_REJECTED = lxnet.Reply("request_rejected", "{}", "", None, None)
_UNAVAILABLE = lxnet.Reply("unavailable", "{}", "", None, None)


class RunnerError(Exception):
    """code: what to report. crashed: the runner died under this load or call. suspect: this
    load or call is the likely cause, because it hung the runner or the runner died while it
    was the only thing to blame."""

    def __init__(self, code, crashed=False, suspect=False):
        super().__init__(code)
        self.code, self.crashed, self.suspect = code, crashed, suspect


class _Waiter:
    """One load or call waiting for Node. Whoever takes it out of its table finishes it."""

    def __init__(self, kind, source_id, generation, timeout):
        self.kind, self.source_id, self.generation = kind, source_id, generation
        self.deadline = time.monotonic() + timeout
        self.value = None
        self.error = None
        self._event = threading.Event()

    def finish(self, value=None, error=None):
        self.value, self.error = value, error
        self._event.set()

    def wait(self, timeout):
        return self._event.wait(max(timeout, 0))


class _Process:
    def __init__(self, popen, generation):
        self.popen = popen
        self.generation = generation
        self.started = time.monotonic()
        self.ready = threading.Event()
        self.is_ready = False
        self.write_lock = threading.Lock()
        self.loaded = set()
        self.exited = False
        self.killed_for = None


def _permission_flag(node):
    """Node 22.13 and later take --permission; earlier 22 releases only --experimental-permission."""
    if not node:
        return None
    for flag in ("--permission", "--experimental-permission"):
        try:
            done = subprocess.run(
                [node, flag, "-e", "0"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, env={}, timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if done.returncode == 0:
            return flag
    return None


def _code(value, default):
    return value if isinstance(value, str) and _CODE.fullmatch(value) else default


def source_url(value, *, trim_first=True):
    """The iOS import URL rule; update alerts measure the raw value before trimming."""
    if not isinstance(value, str):
        return None
    url = value.strip()
    try:
        if len((url if trim_first else value).encode("utf-8")) > 2048:
            return None
        parts = urllib.parse.urlsplit(url)
        if (
            parts.scheme.lower() != "https" or not parts.hostname
            or parts.username is not None or parts.password is not None or "#" in url
        ):
            return None
    except (ValueError, UnicodeError):
        return None
    return url


def update_alert(value):
    """Validate again in Python, the runner protocol's trust boundary."""
    if not isinstance(value, dict) or not isinstance(value.get("log"), str):
        return None
    try:
        if len(json.dumps(value, ensure_ascii=False).encode("utf-8")) > 8 * 1024:
            return None
    except (ValueError, UnicodeError):
        return None
    message = value["log"].strip()[:1024]
    if not message:
        return None
    return message, source_url(value.get("updateUrl"), trim_first=False)


def _platforms(value):
    """The loaded message's sources: known platforms, each with known tiers in tier order."""
    if not isinstance(value, dict) or not value:
        return None
    result = {}
    for platform, qualities in value.items():
        if platform not in PLATFORMS or not isinstance(qualities, list):
            return None
        if not all(isinstance(quality, str) and quality in QUALITIES for quality in qualities):
            return None
        result[platform] = [quality for quality in QUALITIES if quality in qualities]
    return result


class Runner:
    """One Node process holding every loaded source.

    load, unload and call are safe from any thread. A watchdog kills Node when a load or a call
    goes unanswered past its deadline (a script can block Node's event loop from a callback,
    where no vm timeout reaches); every waiting load and call then fails and Node is restarted,
    after which on_ready runs so the enabled sources can be loaded again.
    """

    def __init__(self, node=None, allow_private=False, load_timeout=LOAD_TIMEOUT, call_timeout=CALL_TIMEOUT):
        self._node = node if node is not None else shutil.which("node")
        self._allow_private = allow_private
        self._load_timeout, self._call_timeout = load_timeout, call_timeout
        self.on_ready = None
        self.on_alert = None
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._load_lock = threading.Lock()
        self._flags = None
        self._current = None
        self._generation = 0
        self._loads = {}
        self._calls = {}
        self._requests = {}
        self._active = collections.Counter()
        self._epochs = collections.Counter()
        self._restarts = 0
        self._next_start = 0.0
        self._closed = False
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=16, thread_name_prefix="lxnet")

    @property
    def running(self):
        with self._lock:
            return self._current is not None

    @property
    def pid(self):
        with self._lock:
            return self._current.popen.pid if self._current is not None else None

    def loaded(self):
        """The sources loaded in the running Node."""
        with self._lock:
            return frozenset(self._current.loaded) if self._current is not None else frozenset()

    def start(self):
        """Starts Node and its supervisor; False when Node or its permission model is missing."""
        flag = _permission_flag(self._node)
        if flag is None:
            LOGGER.warning("LXRUNNER unavailable reason=%s", "no_permission_model" if self._node else "node_missing")
            return False
        self._flags = (flag, f"--allow-fs-read={RUNNER_DIR}/", f"--max-old-space-size={HEAP_MB}")
        self._start_process()
        threading.Thread(target=self._supervise, name="lxrunner", daemon=True).start()
        return True

    def close(self):
        with self._wake:
            self._closed = True
            process = self._current
            self._wake.notify_all()
        if process is not None:
            self._kill(process)
        self._pool.shutdown(wait=False, cancel_futures=True)

    def load(self, source_id, script, meta):
        """Loads or reloads one source; returns {platform: [tiers]} from its init."""
        with self._load_lock:
            with self._lock:
                process = self._current
                if process is None:
                    raise RunnerError("runner_unavailable")
                self._epochs[source_id] += 1
                process.loaded.discard(source_id)
                aborts = self._aborts_for(source_id)
                waiter = _Waiter("load", source_id, process.generation, self._load_timeout)
                self._loads[source_id] = waiter
            for abort in aborts:
                abort.abort("cancelled")
            self._send(process, {
                "type": "load", "id": source_id, "script": script, "meta": meta,
                "timeoutMs": round(self._load_timeout * 1000),
            })
            return self._wait(waiter, self._loads, source_id)

    def unload(self, source_id):
        with self._lock:
            process = self._current
            self._epochs[source_id] += 1
            waiter = self._loads.pop(source_id, None)
            if waiter is not None:
                waiter.finish(error=RunnerError("cancelled"))
            aborts = self._aborts_for(source_id)
            if process is not None:
                process.loaded.discard(source_id)
        for abort in aborts:
            abort.abort("cancelled")
        if process is not None:
            self._send(process, {"type": "unload", "id": source_id})

    def call(self, source_id, platform, quality, music_info):
        """One musicUrl; returns the address the script answered with, unchecked."""
        call_id = uuid.uuid4().hex
        with self._lock:
            process = self._current
            if process is None:
                raise RunnerError("runner_unavailable")
            if source_id not in process.loaded:
                raise RunnerError("unavailable")
            waiter = _Waiter("call", source_id, process.generation, self._call_timeout)
            self._calls[call_id] = waiter
        self._send(process, {
            "type": "call", "callId": call_id, "id": source_id, "source": platform, "quality": quality,
            "musicInfo": music_info, "timeoutMs": round(self._call_timeout * 1000),
        })
        return self._wait(waiter, self._calls, call_id)

    def _wait(self, waiter, table, key):
        # The watchdog answers a hung Node long before this; the limit only keeps a thread from
        # waiting forever should the watchdog itself be gone.
        if not waiter.wait(waiter.deadline - time.monotonic() + GRACE + 10):
            with self._lock:
                if table.get(key) is waiter:
                    del table[key]
                    waiter.finish(error=RunnerError("timeout" if waiter.kind == "call" else "initialization_timeout"))
        if waiter.error is not None:
            raise waiter.error
        return waiter.value

    def _aborts_for(self, source_id):
        return [abort for (_, owner, _), abort in self._requests.items() if owner == source_id]

    def _send(self, process, message):
        line = json.dumps(message, ensure_ascii=True, separators=(",", ":")).encode("ascii") + b"\n"
        with process.write_lock:
            try:
                process.popen.stdin.write(line)
                process.popen.stdin.flush()
            except (OSError, ValueError):
                # Node is gone; its exit fails whatever waits on it.
                return False
        return True

    @staticmethod
    def _kill(process):
        try:
            process.popen.kill()
        except OSError:
            pass

    def _schedule_restart(self, ran):
        if ran >= STABLE_SECONDS:
            self._restarts = 0
        delay = RESTART_DELAYS[min(self._restarts, len(RESTART_DELAYS) - 1)]
        self._restarts += 1
        self._next_start = time.monotonic() + delay

    def _start_process(self):
        command = [self._node, *self._flags, str(RUNNER_DIR / "runner.mjs")]
        try:
            popen = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env={}, cwd=RUNNER_DIR, close_fds=True,
            )
        except OSError:
            LOGGER.warning("LXRUNNER start-failed reason=spawn")
            with self._lock:
                self._schedule_restart(0)
            return False
        with self._lock:
            self._generation += 1
            process = _Process(popen, self._generation)
        threading.Thread(target=self._read_stdout, args=(process,), daemon=True).start()
        threading.Thread(target=self._read_stderr, args=(process,), daemon=True).start()
        # Set by the ready line, or by the exit of a Node that never got that far.
        process.ready.wait(READY_TIMEOUT)
        with self._lock:
            started = process.is_ready and not process.exited and not self._closed
            if started:
                self._current = process
            elif not self._closed:
                self._schedule_restart(0)
            closed = self._closed
        if not started:
            self._kill(process)
            if not closed:
                LOGGER.warning("LXRUNNER start-failed reason=%s", "exited" if process.exited else "not_ready")
            return False
        LOGGER.info("LXRUNNER started generation=%s", process.generation)
        callback = self.on_ready
        if callback is not None:
            threading.Thread(target=callback, name="lxrunner-ready", daemon=True).start()
        return True

    def _supervise(self):
        while True:
            with self._wake:
                if not self._closed:
                    self._wake.wait(0.5)
                if self._closed:
                    return
                process = self._current
                now = time.monotonic()
                hung = []
                if process is not None and process.killed_for is None:
                    hung = [
                        waiter for waiter in (*self._loads.values(), *self._calls.values())
                        if waiter.generation == process.generation and now > waiter.deadline + GRACE
                    ]
                    if hung:
                        process.killed_for = set(hung)
                start = process is None and now >= self._next_start
            if hung:
                LOGGER.warning("LXRUNNER hung source=%s kind=%s", hung[0].source_id, hung[0].kind)
                self._kill(process)
            elif start:
                self._start_process()

    def _exited(self, process):
        try:
            code = process.popen.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._kill(process)
            code = process.popen.wait()
        with self._wake:
            process.exited = True
            process.ready.set()
            was_current = self._current is process
            if was_current:
                self._current = None
                self._schedule_restart(time.monotonic() - process.started)
            waiters = []
            for table in (self._loads, self._calls):
                for key in [key for key, waiter in table.items() if waiter.generation == process.generation]:
                    waiters.append(table.pop(key))
            aborts = [abort for (generation, _, _), abort in self._requests.items() if generation == process.generation]
            killed_for = process.killed_for or set()
            closed = self._closed
            for waiter in waiters:
                if waiter in killed_for:
                    code_name = "timeout" if waiter.kind == "call" else "initialization_timeout"
                    waiter.finish(error=RunnerError(code_name, crashed=True, suspect=True))
                else:
                    waiter.finish(error=RunnerError("runner_crashed", crashed=True, suspect=not killed_for))
            self._wake.notify_all()
        for abort in aborts:
            abort.abort("cancelled")
        if was_current and not closed:
            LOGGER.warning("LXRUNNER exit code=%s waiting=%s", code, len(waiters))

    def _read_stdout(self, process):
        stream = process.popen.stdout
        try:
            while True:
                line = stream.readline(MAX_LINE_BYTES + 1)
                if not line:
                    break
                if len(line) > MAX_LINE_BYTES and not line.endswith(b"\n"):
                    while line and not line.endswith(b"\n"):
                        line = stream.readline(MAX_LINE_BYTES + 1)
                    LOGGER.warning("LXRUNNER oversized-line")
                    continue
                try:
                    message = json.loads(line)
                except ValueError:
                    LOGGER.warning("LXRUNNER bad-line")
                    continue
                if isinstance(message, dict):
                    try:
                        self._dispatch(process, message)
                    except Exception as error:
                        LOGGER.warning("LXRUNNER dispatch-error type=%s", type(error).__name__)
        except (OSError, ValueError):
            pass
        finally:
            self._exited(process)

    def _read_stderr(self, process):
        # Only the runner's own LXRUNNER lines and V8's fatal errors are logged; anything else
        # is counted, never shown, in case it quotes a script.
        window, shown, suppressed, other = time.monotonic(), 0, 0, 0
        try:
            for raw in iter(lambda: process.popen.stderr.readline(4096), b""):
                text = raw.decode("utf-8", "replace").strip()
                if text.startswith("LXRUNNER "):
                    text = text[len("LXRUNNER "):]
                elif text.startswith("FATAL ERROR:"):
                    text = "node-fatal " + text[len("FATAL ERROR:"):].strip()
                else:
                    other += 1
                    continue
                now = time.monotonic()
                if now - window >= 60:
                    if suppressed:
                        LOGGER.warning("LXRUNNER suppressed=%s", suppressed)
                    window, shown, suppressed = now, 0, 0
                if shown < STDERR_LINES_PER_MINUTE:
                    shown += 1
                    LOGGER.warning("LXRUNNER %s", _UNPRINTABLE.sub("?", text)[:200])
                else:
                    suppressed += 1
        except (OSError, ValueError):
            pass
        if suppressed:
            LOGGER.warning("LXRUNNER suppressed=%s", suppressed)
        if other:
            LOGGER.warning("LXRUNNER stderr-other lines=%s", other)

    def _dispatch(self, process, message):
        kind = message.get("type")
        if kind == "ready":
            process.is_ready = True
            process.ready.set()
        elif kind == "loaded":
            self._finish_load(process, message)
        elif kind == "result":
            self._finish_call(process, message)
        elif kind == "request":
            self._start_request(process, message)
        elif kind == "cancel":
            slot = (process.generation, message.get("id"), message.get("requestKey"))
            with self._lock:
                abort = self._requests.get(slot)
            if abort is not None:
                abort.abort("cancelled")
        elif kind == "alert":
            source_id = message.get("id")
            alert = update_alert(message)
            if isinstance(source_id, str) and _SOURCE_ID.fullmatch(source_id) and alert is not None:
                callback = self.on_alert
                if callback is not None:
                    # A load waits for this reader. Persist its alert after the store's
                    # current operation, without holding up the loaded response.
                    self._submit(callback, source_id, *alert)

    def _finish_load(self, process, message):
        source_id = message.get("id")
        stray = False
        with self._lock:
            waiter = self._loads.get(source_id) if isinstance(source_id, str) else None
            if waiter is None or waiter.generation != process.generation:
                return
            del self._loads[source_id]
            if message.get("ok") is True:
                platforms = _platforms(message.get("sources"))
                if platforms is None:
                    stray = True
                    waiter.finish(error=RunnerError("protocol_violation"))
                else:
                    process.loaded.add(source_id)
                    waiter.finish(value=platforms)
            else:
                waiter.finish(error=RunnerError(_code(message.get("error"), "initialization_failed")))
        if stray:
            self._submit(self._send, process, {"type": "unload", "id": source_id})

    def _finish_call(self, process, message):
        call_id = message.get("callId")
        with self._lock:
            waiter = self._calls.get(call_id) if isinstance(call_id, str) else None
            if waiter is None or waiter.generation != process.generation:
                return
            del self._calls[call_id]
            if message.get("ok") is True and isinstance(message.get("url"), str):
                waiter.finish(value=message["url"])
            else:
                waiter.finish(error=RunnerError(_code(message.get("error"), "invalid_result")))

    def _submit(self, function, *args):
        try:
            self._pool.submit(function, *args)
        except RuntimeError:
            pass

    def _start_request(self, process, message):
        source_id, key = message.get("id"), message.get("requestKey")
        if not isinstance(source_id, str) or not isinstance(key, str):
            return
        slot = (process.generation, source_id, key)
        with self._lock:
            busy = slot in self._requests or self._active[slot[:2]] >= MAX_REQUESTS_PER_SOURCE
            if not busy:
                abort = lxnet.Abort()
                self._requests[slot] = abort
                self._active[slot[:2]] += 1
                epoch = self._epochs[source_id]
        # Never write to Node from this thread: Node may be blocked writing to us.
        if busy:
            self._submit(self._respond, process, source_id, key, _REJECTED)
        else:
            self._submit(self._perform, process, slot, message.get("url"), message.get("options"), abort, epoch)

    def _perform(self, process, slot, url, options, abort, epoch):
        _, source_id, key = slot
        reply = _UNAVAILABLE
        try:
            reply = lxnet.perform(url, options, allow_private=self._allow_private, abort=abort)
        except Exception as error:
            LOGGER.warning("LXRUNNER request-error type=%s", type(error).__name__)
        finally:
            with self._lock:
                self._requests.pop(slot, None)
                self._active[slot[:2]] -= 1
                if self._active[slot[:2]] <= 0:
                    del self._active[slot[:2]]
                wanted = self._current is process and self._epochs[source_id] == epoch
        if reply.error not in ("", "cancelled") or (reply.status or 0) >= 400:
            LOGGER.info(
                "LXRUNNER request host=%s code=%s status=%s",
                reply.host or "unknown", reply.error or "ok", reply.status or "-",
            )
        # Node waits for an answer even to a cancelled request; it tells the script.
        if wanted:
            self._respond(process, source_id, key, reply)

    def _respond(self, process, source_id, key, reply):
        self._send(process, {
            "type": "response", "id": source_id, "requestKey": key, "errorCode": reply.error,
            "responseJSON": reply.response_json, "bodyText": reply.body_text,
        })
