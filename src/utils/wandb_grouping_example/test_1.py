import wandb
import random
import numpy as np


for j in range(5):
    wandb.init(
        project='wandb_grouping_example-grouping',
        name=f"arg-{j}",
        group="noraml",
        tags=["2"],
        reinit=True,
    )
    for j in range(500):
        wandb.log({"normal": random.normalvariate(0.5, 10)})

wandb.finish()
