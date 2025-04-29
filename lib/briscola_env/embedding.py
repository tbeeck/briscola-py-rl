import numpy as np
from typing import List
from lib.briscola.briscola import BriscolaDeck
from lib.briscola.game import BriscolaGame, BriscolaCard

# Our hand (40)
# The trick so far (3)
# The Briscola (1)
# Unseen cards (40)
# our points (1)
EMBEDDING_SHAPE = (40 + 3 + 1 + 1 + 1 + 40,)


def game_embedding(game: BriscolaGame, player: int):
    full_embeddings = np.zeros(shape=EMBEDDING_SHAPE, dtype=np.uint8)
    offset = 0

    # hand
    for i, v in enumerate(full_cards_embedding(game.players[player].hand)):
        full_embeddings[i + offset] = v
    offset += 40

    # trick
    for i, v in enumerate(cards_embedding(game.trick, 3)):
        full_embeddings[i + offset] = v
    offset += 3

    # trick length
    full_embeddings[offset] = len(game.trick)
    offset += 1

    # briscola
    full_embeddings[offset] = card_embedding(game.briscola)
    offset += 1

    # briscola suit
    briscola_suit_index = BriscolaCard.SUITS.index(game.briscola.suit)
    full_embeddings[offset] = briscola_suit_index
    offset += 1

    # deck + other players hands (unaccounted for cards)
    for i, v in enumerate(remaining_card_embedding(game)):
        full_embeddings[i + offset] = v
    offset += 40

    return np.array(full_embeddings, dtype=np.uint8)


def remaining_card_embedding(game: BriscolaGame):
    existing_cards = list(game.deck.cards)
    for player in game.players:
        existing_cards += list(player.hand)
    return full_cards_embedding(existing_cards) 


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

def full_cards_embedding(cards: list[BriscolaCard]):
    result = np.zeros(shape=(40,), dtype=int)
    have_cards = set(card_embedding(c) for c in cards)
    for i in range(40):
        if i in have_cards:
            result[i] = 1
    return result
