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

    def step(self, action):
        agent = self.agent_selection
        played_card = card_reverse_embedding(action)

        self.game.play(played_card)
        if self.game.should_score_trick():
            self.game.score_trick()
        if self.game.needs_redeal():
            self.game.redeal()

        self.agent_selection = self.agents[
            self.agents.index(agent) + 1 % len(self.agents)
        ]

    def reset(self, seed=None, options=None):
        self.game = BriscolaGame(players=4, goes_first=0, seed=seed)
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.observations = {agent: self.observe(agent) for agent in self.agents}
        self.agent_selection = self.agents[0]
        self._cumulative_rewards = {agent: 0 for agent in self.agents}

    def observe(self, agent):
        agent_id = player_id(agent)
        player_hand = self.game.players[agent_id].hand
        hand_mask = [
            1 if card_reverse_embedding(i) in player_hand else 0 for i in range(40)
        ]
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
        return spaces.Box(
            low=0, high=255, shape=(3 + 3 + 1 + 40 + 1 + 3,), dtype=np.uint8
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # 40 possible cards to play
        # need to mask the space to the available cards in hand
        return spaces.Discrete(40)

    def render(self):
        print(self.game)
