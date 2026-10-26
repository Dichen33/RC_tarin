from __future__ import annotations

import sys
from pathlib import Path


MJLAB_ROOT = Path(__file__).resolve().parents[2]
if str(MJLAB_ROOT) not in sys.path:
    sys.path.insert(0, str(MJLAB_ROOT))

from minidog_rl.scripts.train_sb3 import main


if __name__ == "__main__":
    main()

