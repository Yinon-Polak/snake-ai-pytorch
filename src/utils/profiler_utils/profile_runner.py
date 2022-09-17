import cProfile
from datetime import datetime
from src.agent import train

if __name__ == '__main__':
    name = 'base-line-to-torch-cat'
    agent_kwargs = {"max_games": 1500}
    timestamp = datetime.now()
    output_dir = '/Users/yinonpolak/Library/CloudStorage/GoogleDrive-yinonpolak@gmail.com/My Drive/projects/snake-ai-pytorch/src/outputs'
    output_path = f'{output_dir}/restats-{name}-{timestamp}'
    cProfile.run('train("split-collision", 0, agent_kwargs={"n_features": 11}, note=None, wandb_mode="disabled",)', output_path)
