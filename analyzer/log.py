import logging
import os
import platform
import sys
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path

from . import ANALYZER_VERSION


LOGGER = logging.getLogger("xiyue")


def setup(data):
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    if LOGGER.handlers:
        return
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stdout = logging.StreamHandler(sys.stdout)
    stdout.setFormatter(formatter)
    LOGGER.addHandler(stdout)
    try:
        directory = Path(data) / "logs"
        directory.mkdir(parents=True, exist_ok=True)
        file = RotatingFileHandler(
            directory / "container.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8",
        )
    except OSError:
        LOGGER.warning("日志文件无法写入，仅输出到标准输出")
    else:
        file.setFormatter(formatter)
        LOGGER.addHandler(file)


def tail(data, lines=300):
    lines = min(2000, max(0, lines))
    try:
        with (Path(data) / "logs" / "container.log").open(encoding="utf-8", errors="replace") as file:
            return [line.rstrip("\r\n") for line in deque(file, maxlen=lines)]
    except OSError:
        return []


def export(data):
    parts = []
    for name in ("container.log.2", "container.log.1", "container.log"):
        try:
            parts.append((Path(data) / "logs" / name).read_bytes())
        except OSError:
            continue
    return b"".join(parts)


def environment_lines(
    cpuinfo="/proc/cpuinfo", meminfo="/proc/meminfo",
    memory_max="/sys/fs/cgroup/memory.max",
    memory_limit="/sys/fs/cgroup/memory/memory.limit_in_bytes",
):
    def value(read):
        try:
            return read() or "未知"
        except Exception:
            return "未知"

    def fields(path):
        result = {}
        try:
            for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
                if ":" in line:
                    key, text = line.split(":", 1)
                    result.setdefault(key.strip(), text.strip())
        except Exception:
            pass
        return result

    cpu = fields(cpuinfo)
    flags = cpu.get("flags", cpu.get("Features"))
    instructions = []
    for name, aliases in (("avx2", ("avx2",)), ("avx", ("avx",)), ("sse4_2", ("sse4_2",)), ("neon", ("neon", "asimd"))):
        present = "未知" if flags is None else "有" if any(flag in flags.split() for flag in aliases) else "无"
        instructions.append(f"{name}={present}")
    memory = fields(meminfo)
    limit = None
    for path in (memory_max, memory_limit):
        try:
            limit = Path(path).read_text(encoding="utf-8").strip()
            break
        except Exception:
            continue
    return [
        f"版本={value(lambda: os.environ.get('TASTE_VERSION', 'dev'))} 分析版本={ANALYZER_VERSION}",
        f"架构={value(platform.machine)} CPU数量={value(os.cpu_count)}",
        f"CPU型号={cpu.get('model name') or cpu.get('Hardware') or cpu.get('CPU part') or '未知'}",
        "指令集 " + " ".join(instructions),
        "内存 MemTotal=" + value(lambda: f"{int(memory['MemTotal'].split()[0]) // 1024} MB")
        + " MemAvailable=" + value(lambda: f"{int(memory['MemAvailable'].split()[0]) // 1024} MB"),
        "容器内存上限=" + value(
            lambda: "无限制" if limit == "max" or int(limit) > 1024 ** 4 else f"{int(limit) // (1024 ** 2)} MB"
        ),
    ]
