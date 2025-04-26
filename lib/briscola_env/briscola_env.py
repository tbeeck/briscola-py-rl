import functools
import numpy as np
from gymnasium import spaces

from lib.briscola.game import BriscolaGame
from lib.briscola_env.embedding import (
    card_reverse_embedding,
    game_embedding,
)
from pettingzoo import AECEnv


def player_id(player: str) -> int:
    return int(player.replace("player_", ""))


class BriscolaEnv(AECEnv):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self):
        super().__init__()
        self.possible_agents = [f"player_{i}" for i in range(4)]

    def reset(self, seed=None, options=None):
        self.game = BriscolaGame(players=4, goes_first=0, seed=seed)
        self.agents = self.possible_agents[:]
        self.observations = {agent: self.observe(agent) for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}

        self.agent_selection = self.agents[0]

    def step(self, action):
        agent = self.agent_selection
        next_agent = self.agents[(self.agents.index(agent) + 1) % len(self.agents)]
        played_card = card_reverse_embedding(action)

        self.game.play(played_card)
        if self.game.should_score_trick():
            next_id = self.game.score_trick()
            next_agent = f"player_{next_id}"
        if self.game.needs_redeal():
            self.game.redeal()
        if self.game.game_over():
            placements = self.game.leaders()
            reward = 3
            for score, id in placements:
                a = f"player_{id}"
                self.rewards[a] = reward
                reward -= 1
                self.terminations[a] = True
            self.agent_seletion = None
            return

        self.observations = {agent: self.observe(agent) for agent in self.agents}
        self.agent_selection = next_agent

    def observe(self, agent):
        agent_id = player_id(agent)
        player_hand = self.game.players[agent_id].hand

        hand_mask = np.zeros(40, dtype=np.int8)
        for i in range(40):
            if card_reverse_embedding(i) in player_hand:
                hand_mask[i] = 1
        return {
            "observation": game_embedding(self.game, agent_id),
            "action_mask": np.array(hand_mask, dtype=np.int8),
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Tips on observation space embedding:
        # https://rlcard.org/games.html
        # Our hand (3)
        # The trick so far (3)
        # The Briscola (1)
        # Cards already played (40)
        # our points (1)
        # opponent points (3)
        return spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0, high=255, shape=(3 + 3 + 1 + 40 + 1 + 3,), dtype=np.uint8
                ),
                "action_mask": spaces.Box(low=0, high=1, shape=(40,), dtype=np.int8),
            }
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # 40 possible cards to play
        # need to mask the space to the available cards in hand
        return spaces.Discrete(40)

    def render(self):
        print(self.game)
