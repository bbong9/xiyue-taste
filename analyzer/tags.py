import re
import unicodedata
from pathlib import Path

import mutagen
from mutagen.id3 import ID3


def lyrics_language(lyrics):
    if lyrics is None:
        return None
    text = re.sub(r"\[\d{1,2}:\d{2}(?:\.\d+)?\]", "", lyrics).strip()
    if not text:
        return None
    if re.search(r"[\u3040-\u30ff\u31f0-\u31ff\uff66-\uff9f]", text):
        return "ja"
    if re.search(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]", text):
        return "ko"
    if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text):
        return "zh"
    letters = [character for character in text if character.isalpha()]
    latin = sum("LATIN" in unicodedata.name(character, "") for character in letters)
    if letters and latin / len(letters) >= 0.5:
        return "en"
    return "unknown"


def _values(metadata, *keys):
    for key in keys:
        value = metadata.get(key)
        if value is None:
            continue
        value = getattr(value, "text", value)
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        return str(value).split("\x00")
    return []


def _first(metadata, *keys):
    values = _values(metadata, *keys)
    return values[0] if values else None


def _lyrics(path, metadata):
    sidecar = path.with_suffix(".lrc")
    if sidecar.is_file():
        lyrics = sidecar.read_text(encoding="utf-8-sig", errors="replace")
        if lyrics.strip():
            return lyrics
    if isinstance(metadata, ID3):
        frames = metadata.getall("USLT")
        if frames:
            return "\n".join(frame.text for frame in frames)
    return _first(metadata, "LYRICS", "\xa9lyr")


def read_tags(path):
    path = Path(path)
    audio = mutagen.File(path)
    metadata = audio.tags if audio is not None and audio.tags is not None else {}
    year = _first(metadata, "date", "year", "TDRC", "TYER", "\xa9day", "WM/Year")
    return {
        "title": _first(metadata, "title", "TIT2", "\xa9nam", "Title") or path.stem,
        "artists": _values(metadata, "artist", "artists", "TPE1", "\xa9ART", "Author")
        or None,
        "album": _first(metadata, "album", "TALB", "\xa9alb", "WM/AlbumTitle"),
        "genre": _first(metadata, "genre", "TCON", "\xa9gen", "WM/Genre"),
        "year": year.split("-", 1)[0] if year is not None else None,
        "lyricsLanguage": lyrics_language(_lyrics(path, metadata)),
    }
