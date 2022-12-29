import wandb

from src.agent import train
from src.collision_type import CollisionType

if __name__ == '__main__':

    params = {
        'starting_epsilon': {'distribution': 'int_uniform', 'min': 800, 'max': 8000, },
        'lr': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e-2, },
    }

    method = "bayes"

    sweep_config = {
        'method': method,
        'metric': {
            'name': 'ma_1000_score',
            'goal': 'maximize'
        },
        'parameters': params,
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 15000,
            'eta': 100,
        }
    }

    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, function=train, count=20)
