"""CLI wrapper for the canonical difficulty ladder evaluator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.game_ladder import main


if __name__ == "__main__":
    main()
