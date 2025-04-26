import gymnasium as gym
import numpy as np
from gymnasium import spaces

from lib.briscola.game import BriscolaGame
from lib.briscola_env.embedding import card_reverse_embedding, game_embedding


class BriscolaEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self):
        super().__init__()
        # 40 possible cards to play
        # need to mask the space to the available cards in hand
        self.action_space = spaces.Discrete(40)
        # Tips on observation space embedding:
        # https://rlcard.org/games.html
        # Our hand (3)
        # The trick so far (3)
        # The Briscola (1)
        # Cards already played (40)
        # our points (1)
        # opponent points (3)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(3 + 3 + 1 + 40 + 1 + 3,), dtype=np.uint8
        )

    def step(self, action):
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
        observation, info = self.observe()

        return observation, reward, terminated, {}, info

    def reset(self, seed=None, options=None):
        self.game = BriscolaGame(players=4, goes_first=0, seed=seed)
        observation, info = self.observe()
        return observation, info

    def render(self):
        print(self.game)

    def observe(self):
        observation = game_embedding(self.game)
        return observation, {}
