import numpy as np
from numpy.testing import assert_array_equal

from lib.briscola.briscola import BriscolaCard, BriscolaDeck
from lib.briscola.game import BriscolaGame
from lib.briscola_env.embedding import (
    full_cards_embedding,
    game_embedding,
    cards_embedding,
    card_embedding,
    card_reverse_embedding,
)


def test_game_embedding():
    # Make games and embed them all hope for no errors
    for _ in range(10_000):
        game = BriscolaGame(4)
        game_embedding(game, 0)


def test_card_embeddings():
    for card in BriscolaDeck().cards:
        embedding = card_embedding(card)
        rev = card_reverse_embedding(embedding)
        assert card == rev


def test_unique_embeddings():
    embeddings = cards_embedding(BriscolaDeck().cards, 40)
    assert 40 == len(embeddings)


def test_padding():
    cards = [BriscolaCard("cups", 1), BriscolaCard("cups", 2)]
    result = cards_embedding(cards, 3)
    assert_array_equal(np.array([0, 1, 0]), result)


def test_deck_embedding():
    deck = BriscolaDeck([BriscolaCard("cups", 2)])
    expected = np.zeros(shape=(40,), dtype=int)
    expected[1] = 1
    assert_array_equal(expected, full_cards_embedding(deck.cards))

    full_deck = BriscolaDeck()
    assert_array_equal(np.ones(shape=(40,), dtype=int), full_cards_embedding(full_deck.cards))
