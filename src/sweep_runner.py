import wandb

from src.agent import train
from src.collision_type import CollisionType

if __name__ == '__main__':

    params = {
        'collision_types': {'values': [[CollisionType.BOTH], [CollisionType.BODY, CollisionType.BORDER]]},
        'starting_epsilon': {'distribution': 'int_uniform', 'min': 40, 'max': 200, },  # 80,
        'random_scale': {'distribution': 'int_uniform', 'min': 150, 'max': 500, },  # 200,
        'max_update_end_steps': {'distribution': 'int_uniform', 'min': 0, 'max': 100, },  # 0,
        'max_update_start_steps': {'distribution': 'int_uniform', 'min': 0, 'max': 10, },  # 0,
    }

    method = "bayes"

    sweep_config = {
        'method': method,
        'metric': {
            'name': 'ma_1000_score',
            'goal': 'maximize'
        },
        'parameters': params,
        'early_terminate':  {
            'type': 'hyperband',
            'min_iter': 1500,
            'eta': 100,
        }
    }

    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, function=train, count=25)
