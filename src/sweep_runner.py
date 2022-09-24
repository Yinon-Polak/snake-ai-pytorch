import wandb

from src.agent import train
from src.collision_type import CollisionType

if __name__ == '__main__':

    params = {
        'max_games': {'values': [3500]},
        'epsilon': {'values': [0]},
        'gamma': {'values': [0.9]},
        'lr': {'values': [0.001]},
        'batch_size': {'values': [1_000]},
        'max_memory': {'values': [100_000]},
        'n_steps_collision_check': {'values': [0, 1, 2, 4, 8]},
        'max_update_steps': {'values': [0, 10, 30, 90, 270]},
        'collision_types': {'values': [[CollisionType.BOTH], [CollisionType.BODY, CollisionType.BORDER]]},
        'model_hidden_size_l1': {'values': [128, 256, 512, 1024]},
    }

    method = "random"

    sweep_config = {
        'method': method,
        'metric': {
            'name': 'mean_score',
            'goal': 'maximize'
        },
        'parameters': params,
        'early_terminate':  {
            'type': 'hyperband',
            'min_iter': 800,
            'eta': 100,
        }
    }

    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, function=train)