import wandb

from src.agent import Agent
from src.game import SnakeGameAI


def train(
        name: str,
        run: int,
        wandb_setttings: wandb.Settings = None,
        agent_kwargs: dict = {},
):
    total_score = 0
    record = 0
    agent = Agent(**agent_kwargs)
    game = SnakeGameAI()

    wandb.init(
        project='sanke-ai',
        name=f"{name}-{run}",
        config={
            "architecture": "Linear_QNet",
            "learning_rate": agent.lr,
            "batch_size": agent.batch_size,
            "max_memory": agent.max_memory,
            "gamma": agent.gamma,
            # "epochs": 10,
        },
        settings=wandb_setttings,
        mode="disabled",
    )
    wandb.bwatch(agent.model)

    while agent.n_games < agent.max_games:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory
            game.reset()
            agent.n_games += 1
            # agent.update_rewards(game, reward)
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            total_score += score
            mean_score = total_score / agent.n_games

            # weights and baises logging
            wandb.log({
                'score': score,
                'mean_score': mean_score,
            })
    wandb.finish()


if __name__ == '__main__':
    agent_kwargs = {"max_games": 50}
    for i in range(3):
        train('split-collision', i, agent_kwargs=agent_kwargs)