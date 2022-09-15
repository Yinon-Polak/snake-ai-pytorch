import pstats
from pathlib import Path
from pstats import SortKey
ststs_dir = Path("/src/outputs")
p = pstats.Stats(str((ststs_dir / "restats-2022-09-14 20:36:35.974953")))
print(p.strip_dirs().sort_stats('time').print_stats())