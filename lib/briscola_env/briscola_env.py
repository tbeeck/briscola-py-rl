import functools
import numpy as np
from gymnasium import spaces

from lib.briscola.game import BriscolaGame
from lib.briscola_env.embedding import (
    EMBEDDING_SHAPE,
    card_embedding,
    card_reverse_embedding,
    game_embedding,
)
from pettingzoo import AECEnv


def player_id(player: str) -> int:
    return int(player.replace("player_", ""))


class BriscolaEnv(AECEnv):
    """Custom Environment that follows gym interface."""

    metadata = {"name": "briscola", "render_modes": ["ansi"], "is_parallelizable": True}

    def __init__(self, num_players=4):
        super().__init__()
        self.num_players = num_players
        self.possible_agents = [f"player_{i}" for i in range(self.num_players)]
        self.render_mode = "ansi"
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
        starting_player = 0 if seed is None else seed % self.num_players
        self.game = BriscolaGame(
            players=self.num_players, goes_first=starting_player, seed=seed
        )
        self.agents = self.possible_agents[:][:self.num_players]
        self.observations = {agent: self.observe(agent) for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {"tricks": 0, "wins": 0} for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}

        self.agent_selection = self.agents[starting_player]

    def set_game_result(self):
        placements = self.game.leaders()
        placement_rewards = [120, 0, 0, 0]
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
        hand_mask = np.zeros(40, dtype=np.int8)
        player_hand = (card_embedding(c) for c in self.game.players[agent_id].hand)
        for c in player_hand:
            hand_mask[c] = 1

        return {
            "observation": game_embedding(self.game, agent_id),
            "action_mask": hand_mask
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        # 40 possible cards to play
        # need to mask the space to the available cards in hand
        return self.action_spaces[agent]

    def action_mask(self):
        return self.observe(self.agent_selection)["action_mask"]
