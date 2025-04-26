import functools
import numpy as np
from gymnasium import spaces

from lib.briscola.game import BriscolaGame
from lib.briscola_env.embedding import card_reverse_embedding, game_embedding
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

        terminated = False
        reward = 0
        if self.game.game_over():
            terminated = True
            if self.game.leaders()[0] == 0:
                reward = 1
        observation, info = self.observe(agent)

        return observation, reward, terminated, {}, info

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

    def render(self):
        print(self.game)

    def observe(self, agent):
        observation = {"observation": game_embedding(self.game, player_id(agent))}
        return observation, {}

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
