import urllib.error
import urllib.request

from .ask import AskError
from .llm import chat_json


ARTISTS_PROMPT = """你是音乐库的歌手名整理助手。下面每行是曲库里的一个歌手名和歌曲数，格式：编号|歌手名|歌曲数。
找出其实是同一位歌手（或同一个组合）的不同写法，比如繁简不同、中英文名、带不带别名、大小写和空格不同。
每组选一个保留的写法：组里有中文名就选中文名（简体优先），否则选歌曲数最多的。保留的写法必须是这一组里原样出现的名字。
不同的人不要合并；组合和成员不要合并；拿不准就不合并。
只输出一个 JSON 对象，不要任何其他文字：{"groups": [{"names": [编号, ...], "to": 编号}]}"""

SONGS_PROMPT = """你是音乐库的标签清理助手。每行是一首歌，格式：编号|歌名|专辑|歌手|文件路径。
找出歌名或专辑里混进去的多余内容并删掉：网址、广告、QQ群、音质标记（无损、FLAC、320k、Hi-Res、24bit 等）、开头的曲目序号、开头重复的歌手名、多余的符号和空格。
歌名或专辑为空、是“未知”或是乱码时，可以从文件名或文件夹名里原样截取正确的名字。
不许删版本信息：Live、现场、伴奏、Remix、粤语版、DJ 版、钢琴版、翻唱等都要保留。
不许翻译、不许改写、不许补你不确定的名字；只能删字，或者从文件路径里原样截取。没问题的歌不要输出，没改的字段不要写。
只输出一个 JSON 对象，不要任何其他文字：{"songs": [{"i": 编号, "title": "改后的歌名", "album": "改后的专辑"}]}"""


class Butler:
    def __init__(self, api_key, base_url, model, timeout=120, urlopen=urllib.request.urlopen):
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._timeout = timeout
        self._urlopen = urlopen

    def artists(self, artists):
        if not self._api_key:
            raise AskError("ask_unconfigured", 503)
        try:
            catalog = "\n".join(f"{i}|{artist['name']}|{artist['songs']}" for i, artist in enumerate(artists))
            result = chat_json(
                self._api_key, self._base_url, self._model, ARTISTS_PROMPT,
                catalog, 2000, self._timeout, self._urlopen,
            )
            if not isinstance(result["groups"], list):
                raise ValueError("Invalid groups")
            used = set()
            groups = []
            for group in result["groups"]:
                if not isinstance(group["names"], list):
                    raise ValueError("Invalid names")
                names = list(dict.fromkeys(
                    i for i in group["names"]
                    if type(i) is int and 0 <= i < len(artists) and i not in used
                ))
                to = group["to"]
                if type(to) is not int or to not in names or len(names) < 2:
                    continue
                used.update(names)
                groups.append({"names": [artists[i]["name"] for i in names], "to": artists[to]["name"]})
            return {"groups": groups}
        except (urllib.error.URLError, OSError, ValueError, TypeError, KeyError, IndexError, AttributeError):
            raise AskError("ask_failed", 502) from None

    def songs(self, songs):
        if not self._api_key:
            raise AskError("ask_unconfigured", 503)
        try:
            catalog = "\n".join("|".join([
                str(i), song.get("title") or "", song.get("album") or "",
                "/".join(song.get("artists") or []), song.get("path") or "",
            ]) for i, song in enumerate(songs))
            result = chat_json(
                self._api_key, self._base_url, self._model, SONGS_PROMPT,
                catalog, 3000, self._timeout, self._urlopen,
            )
            if not isinstance(result["songs"], list):
                raise ValueError("Invalid songs")
            used = set()
            fixes = []
            for item in result["songs"]:
                i = item["i"]
                if type(i) is not int or not 0 <= i < len(songs) or i in used:
                    continue
                used.add(i)
                fix = {"id": songs[i]["id"]}
                for field in ("title", "album"):
                    value = item.get(field)
                    if isinstance(value, str):
                        value = value.strip()
                        if value and len(value) <= 200 and value != songs[i].get(field):
                            fix[field] = value
                if len(fix) > 1:
                    fixes.append(fix)
            return {"songs": fixes}
        except (urllib.error.URLError, OSError, ValueError, TypeError, KeyError, IndexError, AttributeError):
            raise AskError("ask_failed", 502) from None
