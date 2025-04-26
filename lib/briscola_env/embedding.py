import numpy as np
from typing import List
from lib.briscola.briscola import BriscolaDeck
from lib.briscola.game import BriscolaGame, BriscolaCard


# Given a game and the current player, generate an embedding for the model
# Our hand (3)
# The trick so far (3)
# The Briscola (1)
# Cards still in the deck (40)
# our points (1)
# opponent points (3)
def game_embedding(game: BriscolaGame, player: int):
    full_embeddings = np.zeros(shape=(3 + 3 + 1 + 40 + 1 + 3,), dtype=int)
    offset = 0
    # hand
    for i, v in enumerate(cards_embedding(game.players[player].hand, 3)):
        full_embeddings[i + offset] = v
    offset += 3
    # trick
    for i, v in enumerate(cards_embedding(game.trick, 3)):
        full_embeddings[i + offset] = v
    offset += 3
    # briscola
    full_embeddings[offset] = card_embedding(game.briscola)
    offset += 1
    # deck
    for i, v in enumerate(deck_embedding(game.deck)):
        full_embeddings[i + offset] = v
    offset += 40
    # player points
    # TODO always put current player points first,
    # then in order of play after current player
    for i, v in enumerate(game.players):
        full_embeddings[i + offset] = v.score()
    return full_embeddings

def deck_embedding(deck: BriscolaDeck):
    result = np.zeros(shape=(40,), dtype=int)
    existing_cards = set(card_embedding(c) for c in deck.cards)
    for i in range(40):
        if i in existing_cards:
            result[i] = 1
    return result


def cards_embedding(cards: List[BriscolaCard], length: int):
    if len(cards) > length:
        raise Exception("cards exceeded expected length")
    result = np.zeros(shape=(length,), dtype=int)
    for i in range(len(cards)):
        result[i] = card_embedding(cards[i])
    return result


def card_embedding(card: BriscolaCard) -> int:
    offset = BriscolaCard.SUITS.index(card.suit) * 10
    result = (card.rank - 1) + offset
    return result


def card_reverse_embedding(i: int) -> BriscolaCard:
    suit_index = i // 10
    i -= suit_index * 10
    rank = i + 1
    return BriscolaCard(BriscolaCard.SUITS[suit_index], rank)
