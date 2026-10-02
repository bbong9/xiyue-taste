import argparse
import ipaddress
import os
import threading
import time

from .access import AccessSettings
from .ask import Asker
from .butler import Butler
from .downloads import Downloads
from .panel import OutputIndex, PanelState
from .scan import scan
from .serve import make_server
from .settings import LLMSettings


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
        server = make_server(
            args.out, args.port, data=args.data, state=state, index=index, asker=asker, butler=butler, llm=llm,
            downloads=Downloads(args.downloads),
            access=access,
        )
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"Serving on port {args.port}.", flush=True)
        while True:
            state.begin_scan()
            try:
                result = scan(
                    args.music, args.data, args.out,
                    workers=args.workers, progress=state.set_progress,
                )
            except Exception as exception:
                state.fail_scan(exception)
            else:
                state.finish_scan(result["analyzed"], result["tracks"])
            summary = state.snapshot()
            print(
                f"Analyzed {summary['lastAnalyzed']} files; exported {summary['lastExported']} tracks.",
                flush=True,
            )
            interval = args.interval_hours * 3600
            state.set_next(time.time() + interval)
            state.wait_for_next(interval)

    while True:
        result = scan(args.music, args.data, args.out, workers=args.workers)
        print(
            f"Analyzed {result['analyzed']} files; exported {result['tracks']} tracks.",
            flush=True,
        )
        if args.command == "scan":
            return
        time.sleep(args.interval_hours * 3600)


if __name__ == "__main__":
    main()
