import numpy as np
from typing import List


class BriscolaCard:
    """
    Class representing a card in the game of Briscola.
    """

    SUITS = ["cups", "coins", "swords", "batons"]
    RANKS = list(range(1, 11))  # Ranks 1 to 10

    def __init__(self, suit: str, rank: int):
        if suit not in self.SUITS:
            raise ValueError(f"Invalid suit: {suit}")
        if rank not in self.RANKS:
            raise ValueError(f"Invalid rank: {rank}")
        self.suit = suit
        self.rank = rank

    def __repr__(self):
        return f"Card({self.suit}, {self.rank})"

    def score(self) -> int:
        """
        Returns the point value of a card.
        """
        return {1: 11, 3: 10, 10: 4, 9: 3, 8: 2}.get(self.rank, 0)

    def strength(self) -> int:
        """
        Returns the strength of a card, used to determine a trick winner.
        """
        score_value = self.score()
        return score_value + 10 if score_value > 0 else self.rank

    def __str__(self):
        rank_names = {1: "ace", 8: "jack", 9: "knight", 10: "king"}
        return f"({rank_names.get(self.rank, self.rank)}, {self.suit})"

    def __eq__(self, val):
        return self.suit == val.suit and self.rank == val.rank


class BriscolaDeck:
    """
    Class representing a deck of cards.
    """

    def __init__(self, cards=None):
        if cards is None:
            self.cards = self.new_deck()
        else:
            self.cards = cards

    def new_deck(self) -> List[BriscolaCard]:
        """
        Create a new deck of cards, not shuffled.
        """
        return [
            BriscolaCard(suit, rank)
            for suit in BriscolaCard.SUITS
            for rank in BriscolaCard.RANKS
        ]

    def shuffle(self, seed=None):
        """
        Shuffle the deck.
        """
        if seed:
            np.random.default_rng(seed).shuffle(self.cards)
        else:
            np.random.shuffle(self.cards)

    def take(self, n: int) -> List[BriscolaCard]:
        """
        Take a number of cards from the top of the deck.
        """
        taken = self.cards[:n]
        self.cards = self.cards[n:]
        return taken

    def __repr__(self):
        return f"Deck({self.cards})"


class BriscolaPlayer:
    """
    Class representing a player in the game of Briscola.
    """

    def __init__(self):
        self.hand = []
        self.pile = []

    def score(self) -> int:
        """
        Calculate the player's score based on their pile.
        """
        return sum(card.score() for card in self.pile)

    def remove_from_hand(self, card: BriscolaCard):
        """
        Remove a specific card from a player's hand.
        """
        if card in self.hand:
            self.hand.remove(card)

    def take_trick(self, cards: List[BriscolaCard]):
        """
        Add won trick cards to the player's pile.
        """
        self.pile.extend(cards)

    def __repr__(self):
        return f"Player(hand={self.hand}, pile={self.pile})"
