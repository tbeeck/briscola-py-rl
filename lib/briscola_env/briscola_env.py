import functools
import numpy as np
from gymnasium import spaces

from lib.briscola.game import BriscolaGame
from lib.briscola_env.embedding import (
    EMBEDDING_SHAPE,
    card_reverse_embedding,
    game_embedding,
)
from pettingzoo import AECEnv


def player_id(player: str) -> int:
    return int(player.replace("player_", ""))


class BriscolaEnv(AECEnv):
    """Custom Environment that follows gym interface."""

    metadata = {"name": "briscola", "render_modes": ["ansi"], "is_parallelizable": True}

    def __init__(self):
        super().__init__()
        self.max_players = 4
        self.possible_agents = [f"player_{i}" for i in range(self.max_players)]
        self.render_mode = "ansi"
        # Tips on observation space embedding:
        # https://rlcard.org/games.html
        # Our hand (3)
        # The trick so far (3)
        # The Briscola (1)
        # Cards already played (40)
        # our points (1)
        # opponent points (3)
        self.observation_spaces = {
            name: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=255, shape=EMBEDDING_SHAPE, dtype=np.uint8
                    ),
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(40,), dtype=np.int8
                    ),
                }
            )
            for name in self.possible_agents
        }
        self.action_spaces = {
            name: spaces.Discrete(40) for name in self.possible_agents
        }

    def reset(self, seed=None, options=None):
        player_count = (
            self.max_players
            if options is None or options.get("player_count") is None
            else options.get("player_count")
        )
        starting_player = 0 if seed is None else seed % player_count
        self.game = BriscolaGame(
            players=player_count, goes_first=starting_player, seed=seed
        )
        self.agents = self.possible_agents[:][:player_count]
        self.observations = {agent: self.observe(agent) for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {"tricks": 0, "wins": 0} for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}

        self.agent_selection = self.agents[starting_player]

    def set_game_result(self):
        placements = self.game.leaders()
        placement_rewards = [200, 50, 0, 0]
        for i in range(len(placements)):
            _score, p = placements[i]
            if i == 0:
                self.infos[self.agents[p]]["wins"] += 1
            a = self.agents[p]
            self.rewards[a] += placement_rewards[i]
            self.terminations[a] = True

    def step(self, action):
        if self.terminations[self.agent_selection]:
            return self._was_dead_step(action)
        self._cumulative_rewards[self.agent_selection] = 0
        played_card = card_reverse_embedding(action)
        self.game.play(played_card)
        if self.game.should_score_trick():
            trick_points = sum(card.score() for card in self.game.trick)
            winner = self.game.score_trick()
            self.rewards[self.agents[winner]] += trick_points
            self.infos[self.agents[winner]]["tricks"] += 1

        if self.game.needs_redeal():
            self.game.redeal()

        if self.game.game_over():
            self.set_game_result()

        self._accumulate_rewards()
        self.observations = {agent: self.observe(agent) for agent in self.agents}
        self.agent_selection = self.agents[self.game.action_on]

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

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        # 40 possible cards to play
        # need to mask the space to the available cards in hand
        return self.action_spaces[agent]

    def action_mask(self):
        return self.observe(self.agent_selection)["action_mask"]
