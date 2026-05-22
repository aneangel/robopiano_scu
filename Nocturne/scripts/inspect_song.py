from __future__ import annotations

import argparse
import json

from _bootstrap import bootstrap

bootstrap()

from nocturne.rp1m import inspect_song, list_songs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an RP1M song group for Nocturne.")
    parser.add_argument("--rp1m-root", required=True)
    parser.add_argument("--song-name", default=None)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    if args.song_name:
        payload = inspect_song(args.rp1m_root, args.song_name)
    else:
        payload = {"songs": list_songs(args.rp1m_root, limit=int(args.limit))}
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
