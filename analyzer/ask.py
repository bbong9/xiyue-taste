import urllib.error
import urllib.request

from .llm import chat_json

SYSTEM_PROMPT = """你是私人音乐助手。根据用户的要求做三件事：
1. 从给出的曲库里挑最多 10 首最符合的，只能用曲库编号。
2. 另外推荐最多 15 首曲库以外、真实存在、在国内音乐平台能搜到的歌，写准确的歌名和第一位歌手，不要编造。
3. 给 2 到 3 个适合在音乐平台搜索歌单的关键词，每个不超过 10 个字。
如果给了“常听歌手”，推荐要贴近这些口味，但不要只推这些歌手。
BPM 越大节奏越快；响度 LUFS 越接近 0 越响，越小越安静。
只输出一个 JSON 对象，不要任何其他文字：{"reason": "一句不超过 30 字的中文说明", "picks": [编号, ...], "songs": [{"title": "歌名", "artist": "歌手"}], "playlists": ["关键词"]}"""

LIBRARY_PROMPT = """你是私人音乐助手。从给出的曲库里挑最多 10 首符合用户要求的歌，只能用曲库编号。
如果要求点名了歌手，只挑这位歌手的歌；点名了歌名，只挑这首歌。曲库里没有符合的就返回空列表 []，不要拿别的歌凑数。
“常听歌手”只在要求是心情、场景、风格这类时用来参考口味；要求点名了歌手或歌名时忽略它。
BPM 越大节奏越快；响度 LUFS 越接近 0 越响，越小越安静。
只输出一个 JSON 对象，不要任何其他文字：{"reason": "一句不超过 30 字的中文说明", "picks": [编号, ...]}"""

ONLINE_PROMPT = """你是私人音乐助手。根据用户的要求做两件事：
1. 推荐最多 10 首真实存在、在国内音乐平台能搜到的歌，写准确的歌名和第一位歌手，不要编造。
2. 给 2 到 3 个适合在音乐平台搜索歌单的关键词，每个不超过 10 个字。
如果要求点名了歌手，只推荐这位歌手的歌，歌单关键词也围绕这位歌手；这时忽略“常听歌手”。
否则如果给了“常听歌手”，推荐要贴近这些口味，但来自常听歌手的歌最多 3 首，其余必须是别的歌手。
只输出一个 JSON 对象，不要任何其他文字：{"songs": [{"title": "歌名", "artist": "歌手"}], "playlists": ["关键词"]}"""

ASK_PARTS = ("all", "library", "online")


def _picks(result, tracks) -> list[int]:
    return list(dict.fromkeys(
        i for i in result["picks"] if type(i) is int and 0 <= i < len(tracks)
    ))[:20]


def _reason(result) -> str:
    reason = result.get("reason")
    return reason[:60] if isinstance(reason, str) else ""


def _songs(result, limit) -> list[dict]:
    songs = []
    seen = set()
    for song in result.get("songs", []) if isinstance(result.get("songs"), list) else []:
        if not isinstance(song, dict):
            continue
        title, artist = song.get("title"), song.get("artist")
        if not isinstance(title, str) or not isinstance(artist, str):
            continue
        title, artist = title.strip(), artist.strip()
        if not 1 <= len(title) <= 60 or not 1 <= len(artist) <= 60 or (title, artist) in seen:
            continue
        seen.add((title, artist))
        songs.append({"title": title, "artist": artist})
        if len(songs) == limit:
            break
    return songs


def _playlists(result) -> list[str]:
    playlists = []
    for keyword in result.get("playlists", []) if isinstance(result.get("playlists"), list) else []:
        if not isinstance(keyword, str):
            continue
        keyword = keyword.strip()
        if 1 <= len(keyword) <= 20 and keyword not in playlists:
            playlists.append(keyword)
        if len(playlists) == 3:
            break
    return playlists


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

    def status(self):
        return {"configured": bool(self._api_key), "model": self._model}

    def ask(self, query, taste=(), part="all"):
        if not self._api_key:
            raise AskError("ask_unconfigured", 503)
        try:
            taste_line = "常听歌手：" + "、".join(taste) + "\n" if taste else ""
            if part == "online":
                result = chat_json(
                    self._api_key, self._base_url, self._model, ONLINE_PROMPT,
                    taste_line + "要求：" + query, 700, self._timeout, self._urlopen,
                )
                if not isinstance(result, dict):
                    raise ValueError("Invalid result")
                return {
                    "reason": "", "songs": _songs(result, 10),
                    "playlists": _playlists(result), "items": [],
                }
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
                self._api_key, self._base_url, self._model,
                LIBRARY_PROMPT if part == "library" else SYSTEM_PROMPT,
                "曲库（编号|歌名|歌手|风格|语种|BPM|响度LUFS）：\n" + catalog + "\n\n"
                + taste_line + "要求：" + query,
                300 if part == "library" else 1200, self._timeout, self._urlopen,
            )
            if not isinstance(result, dict) or not isinstance(result.get("picks"), list):
                raise ValueError("Invalid picks")
            return {
                "reason": _reason(result),
                "songs": [] if part == "library" else _songs(result, 15),
                "playlists": [] if part == "library" else _playlists(result),
                "items": [
                    {"index": i, "title": tracks[i]["title"],
                     "artists": tracks[i]["artists"], "path": tracks[i]["path"]}
                    for i in _picks(result, tracks)
                ],
            }
        except (urllib.error.URLError, OSError, ValueError, TypeError, KeyError, IndexError, AttributeError):
            raise AskError("ask_failed", 502) from None
