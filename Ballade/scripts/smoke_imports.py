from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ballade import OnlineJacobianTracker, ResidualMLPController, build_micro_q_targets  # noqa: E402


def main() -> None:
    assert build_micro_q_targets
    assert OnlineJacobianTracker
    assert ResidualMLPController
    print("Ballade smoke imports passed")


if __name__ == "__main__":
    main()
