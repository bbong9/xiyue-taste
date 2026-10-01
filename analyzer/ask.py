import urllib.error
import urllib.request

from .llm import chat_json

SYSTEM_PROMPT = """你是私人音乐库的选歌助手。只能从给出的曲库里选，按最符合要求的顺序挑最多 20 首。
BPM 越大节奏越快；响度 LUFS 越接近 0 越响，越小越安静。
只输出一个 JSON 对象，不要任何其他文字：{"reason": "一句不超过 30 字的中文说明", "picks": [编号, ...]}"""


class AskError(Exception):
    def __init__(self, code, status):
        super().__init__(code)
        self.code = code
        self.status = status


class Asker:
    def __init__(self, index, api_key, base_url, model, timeout=60, urlopen=urllib.request.urlopen):
        self._index = index
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._timeout = timeout
        self._urlopen = urlopen

    def ask(self, query):
        if not self._api_key:
            raise AskError("ask_unconfigured", 503)
        try:
            tracks = self._index.load()
            catalog = "\n".join(
                "|".join([
                    str(i),
                    track.get("title") or "",
                    "/".join(track.get("artists") or []),
                    track.get("genre") or "",
                    track.get("lyricsLanguage") or "",
                    f"{track['bpm']:.0f}" if track.get("bpm") is not None else "",
                    f"{track['loudnessLUFS']:.1f}" if track.get("loudnessLUFS") is not None else "",
                ])
                for i, track in enumerate(tracks)
            )
            result = chat_json(
                self._api_key, self._base_url, self._model, SYSTEM_PROMPT,
                "曲库（编号|歌名|歌手|风格|语种|BPM|响度LUFS）：\n" + catalog + "\n\n要求：" + query,
                400, self._timeout, self._urlopen,
            )
            if not isinstance(result, dict) or not isinstance(result.get("picks"), list):
                raise ValueError("Invalid picks")
            picks = list(dict.fromkeys(
                i for i in result["picks"] if type(i) is int and 0 <= i < len(tracks)
            ))[:20]
            reason = result.get("reason")
            reason = reason[:60] if isinstance(reason, str) else ""
            return {
                "reason": reason,
                "items": [
                    {"index": i, "title": tracks[i]["title"],
                     "artists": tracks[i]["artists"], "path": tracks[i]["path"]}
                    for i in picks
                ],
            }
        except (urllib.error.URLError, OSError, ValueError, TypeError, KeyError, IndexError, AttributeError):
            raise AskError("ask_failed", 502) from None
