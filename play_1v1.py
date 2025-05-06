import glob
import os
import time

from pettingzoo.test import api_test
from sb3_contrib import MaskablePPO

from lib.briscola_env.briscola_env import BriscolaEnv
from lib.briscola_env.embedding import card_embedding, card_reverse_embedding


PLAYER_COUNT = 2


def make_env():
    return BriscolaEnv(num_players=PLAYER_COUNT)


api_test(make_env(), num_cycles=1000)


def play_1v1(player_position):
    # Evaluate a trained agent vs a random agent
    env = make_env()
    print("Starting 1v1 game with player at position", player_position)
    try:
        latest_policy = max(
            glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        raise
    print("Using policy:", latest_policy)
    model = MaskablePPO.load(latest_policy)

    env.reset()

    for agent in env.agent_iter():
        print("Rewards:", env.rewards)
        obs, reward, termination, truncation, info = env.last()
        observation, action_mask = obs.values()
        if termination or truncation:
            winner = max(env.rewards, key=env.rewards.get)
            print("Winner:", winner)
            break
        if agent == env.possible_agents[player_position]:
            print("----")
            print(env.game)
            card_idx = (
                int(input(f"Select a card to play (player {player_position}): ")) - 1
            )
            card = env.game.players[player_position].hand[card_idx]
            act = card_embedding(card)
            print("You played", card)
        else:
            act = int(
                model.predict(
                    observation, action_masks=action_mask, deterministic=True
                )[0]
            )
            print("AI played", card_reverse_embedding(act))
        time.sleep(0.2)
        env.step(act)

    env.close()


if __name__ == "__main__":
    play_1v1(0)
