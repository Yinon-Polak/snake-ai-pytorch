import pstats
from pathlib import Path
from pstats import SortKey
ststs_dir = Path("/Users/yinonpolak/Library/CloudStorage/GoogleDrive-yinonpolak@gmail.com/My Drive/projects/snake-ai-pytorch/src/outputs/")
p = pstats.Stats(str((ststs_dir / "restats-base-line-to-torch-cat-2022-09-17 10:52:02.427739")))
print(p.strip_dirs().sort_stats('time').print_stats())