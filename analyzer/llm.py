import json
import urllib.request


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
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        content = json.loads(response.read())["choices"][0]["message"]["content"]
    result = json.loads(content[content.index("{"):content.rindex("}") + 1])
    if not isinstance(result, dict):
        raise ValueError("Invalid object")
    return result
