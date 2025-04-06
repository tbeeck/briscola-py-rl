import gymnasium as gym
import numpy as np
from gymnasium import spaces


class BriscolaEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self):
        super().__init__()
        # 3 cards to play (need to mask in the case that the player has less than 3 cards)
        self.action_space = spaces.Discrete(3)
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
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        return observation, info

    def render(self):
        pass

    def close(self):
        pass
