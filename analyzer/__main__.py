import argparse
import ipaddress
import os
import threading
import time

from . import log
from .access import AccessSettings
from .ask import Asker
from .butler import Butler
from .downloads import Downloads
from .lxhost import Runner
from .panel import OutputIndex, PanelState
from .log import LOGGER
from .resolver import Resolver
from .scan import probe_worker, scan
from .serve import make_server
from .settings import LLMSettings
from .sources import SourceStore


def _run_sources(runner, sources):
    """Starts the source runner; every 10 minutes, enabled sources whose load failed get another try."""
    if not runner.start():
        return
    while True:
        time.sleep(600)
        sources.retry_failed()


def main():
    parser = argparse.ArgumentParser(prog="python -m analyzer")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("scan", "loop"):
        command = commands.add_parser(name)
        command.add_argument("--music", required=True)
        command.add_argument("--data", required=True)
        command.add_argument("--out", required=True)
        command.add_argument("--workers", type=int, default=2)
        if name == "loop":
            command.add_argument("--interval-hours", type=float, default=24)
    command = commands.add_parser("run")
    command.add_argument("--music", required=True)
    command.add_argument("--data", required=True)
    command.add_argument("--out", required=True)
    command.add_argument("--workers", type=int, default=2)
    command.add_argument("--interval-hours", type=float, default=24)
    command.add_argument("--port", type=int, default=8790)
    command.add_argument("--downloads", default="/downloads")
    args = parser.parse_args()

    if args.command == "run":
        log.setup(args.data)
        for line in log.environment_lines():
            LOGGER.info("ENV %s", line)
        try:
            with os.scandir(args.music) as entries:
                music_count = sum(1 for _ in entries)
        except OSError:
            music_count = "未知"
        LOGGER.info(
            "MOUNTS music_read=%s music_entries=%s downloads_exists=%s downloads_write=%s out_write=%s",
            "能" if os.access(args.music, os.R_OK) else "不能", music_count,
            "是" if os.path.exists(args.downloads) else "否",
            "能" if os.access(args.downloads, os.W_OK) else "不能",
            "能" if os.access(args.out, os.W_OK) else "不能",
        )
        network = os.environ.get("TASTE_TRUSTED_NETWORK")
        host_ip = os.environ.get("TASTE_HOST_IP", "")
        try:
            network = ipaddress.ip_network(network, strict=False) if network else None
        except ValueError:
            raise SystemExit("bad TASTE_TRUSTED_NETWORK")
        access = AccessSettings(
            os.path.join(args.data, "access-settings.json"),
            token=os.environ.get("TASTE_ACCESS_TOKEN", ""), network=network, host_ip=host_ip,
        )
        LOGGER.info("SETTINGS access status=%s", access.status())
        LOGGER.info("WORKER-PROBE %s", probe_worker())
        os.nice(15)
        state = PanelState()
        index = OutputIndex(args.out)
        api_key = os.environ.get("TASTE_LLM_API_KEY", "")
        base_url = os.environ.get("TASTE_LLM_BASE_URL", "https://api.siliconflow.cn/v1")
        model = os.environ.get("TASTE_LLM_MODEL", "deepseek-ai/DeepSeek-V3.2")
        asker = Asker(index, api_key, base_url, model)
        butler = Butler(api_key, base_url, model)
        llm = LLMSettings(
            os.path.join(args.data, "llm-settings.json"), api_key, base_url, model, targets=(asker, butler),
        )
        runner = Runner()
        sources = SourceStore(args.data, runner)
        resolver = Resolver(sources, runner, args.data)
        runner.on_ready = sources.reload_all
        threading.Thread(target=_run_sources, args=(runner, sources), name="sources", daemon=True).start()
        server = make_server(
            args.out, args.port, data=args.data, state=state, index=index, asker=asker, butler=butler, llm=llm,
            downloads=Downloads(args.downloads),
            access=access, sources=sources, resolver=resolver,
        )
        threading.Thread(target=server.serve_forever, daemon=True).start()
        LOGGER.info("Serving on port %s.", args.port)
        while True:
            state.begin_scan()
            try:
                result = scan(
                    args.music, args.data, args.out,
                    workers=args.workers, progress=state.set_progress,
                )
            except Exception as exception:
                state.fail_scan(exception)
                LOGGER.exception("SCAN-ABORT %s", type(exception).__name__)
            else:
                state.finish_scan(result["analyzed"], result["tracks"])
            summary = state.snapshot()
            LOGGER.info(
                "Analyzed %s files; exported %s tracks.",
                summary["lastAnalyzed"], summary["lastExported"],
            )
            interval = args.interval_hours * 3600
            state.set_next(time.time() + interval)
            state.wait_for_next(interval)

    while True:
        result = scan(args.music, args.data, args.out, workers=args.workers)
        LOGGER.info(
            "Analyzed %s files; exported %s tracks.",
            result["analyzed"], result["tracks"],
        )
        if args.command == "scan":
            return
        time.sleep(args.interval_hours * 3600)


if __name__ == "__main__":
    main()
