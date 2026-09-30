import gzip
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from . import ANALYZER_VERSION


def write_output(out, tracks, vectors, neighbors):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    document = {
        "schema": 1,
        "analyzerVersion": ANALYZER_VERSION,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "tracks": [
            {
                "path": track["path"],
                "size": track["size"],
                "mtime": track["mtime"],
                **track["tags"],
                "durationSec": track["features"]["durationSec"],
                "bpm": track["features"]["bpm"],
                "loudnessLUFS": track["features"]["loudnessLUFS"],
                "vector": vectors[index].round(4).tolist(),
                "neighbors": neighbors[index],
            }
            for index, track in enumerate(tracks)
        ],
    }
    with tempfile.NamedTemporaryFile(dir=out, suffix=".json.gz", delete=False) as temporary:
        temporary_path = temporary.name
    with gzip.open(temporary_path, "wt", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, allow_nan=False)
    os.replace(temporary_path, out / "xiyue-taste-v1.json.gz")
