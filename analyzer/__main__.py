import argparse
import time

from .scan import scan


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
    args = parser.parse_args()

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
