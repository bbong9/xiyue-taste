import io
import json
import re
import urllib.error
import urllib.request

from .log import LOGGER


# Providers whose models think before answering unless told not to. Thinking
# spends the answer's token budget and makes every request slower, and all we
# want back is a small JSON object.
_NO_THINKING = (
    ("api.deepseek.com", {"thinking": {"type": "disabled"}}),
    ("open.bigmodel.cn", {"thinking": {"type": "disabled"}}),
    ("volces.com", {"thinking": {"type": "disabled"}}),
    ("dashscope.aliyuncs.com", {"enable_thinking": False}),
)


class Truncated(ValueError):
    """The model ran out of tokens before it produced the JSON."""


def _post(api_key, base_url, body, timeout, urlopen):
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def chat_json(api_key, base_url, model, system, user, max_tokens, timeout, urlopen):
    """POST chat/completions and decode the first through last braces as a dict."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }
    extra = next((dict(fields) for host, fields in _NO_THINKING if host in base_url), {})
    try:
        payload = _post(api_key, base_url, {**body, **extra}, timeout, urlopen)
    except urllib.error.HTTPError as error:
        if error.code != 400:
            raise
        try:
            raw = error.read(2000)
        except (OSError, AttributeError):
            raw = b""
        detail = raw.decode("utf-8", "replace")
        retry = dict(body)
        if "max_completion_tokens" in detail or "temperature" in detail:
            # Models that only take the newer names and a fixed temperature.
            retry.pop("temperature")
            retry["max_completion_tokens"] = retry.pop("max_tokens")
        elif not extra:
            raise urllib.error.HTTPError(error.url, error.code, error.msg, error.headers, io.BytesIO(raw)) from None
        LOGGER.info("LLM-RETRY model=%s after=400", model)
        payload = _post(api_key, base_url, retry, timeout, urlopen)
    choice = payload["choices"][0]
    content = choice["message"].get("content") or ""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
    try:
        result = json.loads(content[content.index("{"):content.rindex("}") + 1])
    except ValueError:
        finish = choice.get("finish_reason")
        LOGGER.warning("LLM-BAD-ANSWER model=%s finish=%s content=%r", model, finish, content[:200])
        if finish == "length":
            raise Truncated("truncated") from None
        raise
    if not isinstance(result, dict):
        raise ValueError("Invalid object")
    return result
