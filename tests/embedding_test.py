import numpy as np
from numpy.testing import assert_array_equal

from lib.briscola.briscola import Card, Deck
from lib.briscola.game import Game
from lib.briscola_env.embedding import (
    game_embedding,
    cards_embedding,
    card_embedding,
    card_reverse_embedding,
    deck_embedding,
)

def test_game_embedding():
    # Make games and embed them all hope for no errors
    for _ in range(10_000):
        game = Game(4)
        _game_embedding(game, 0)

def test_card_embeddings():
    for card in Deck().cards:
        embedding = card_embedding(card)
        rev = card_reverse_embedding(embedding)
        print(card, embedding)
        assert card == rev


def test_unique_embeddings():
    embeddings = cards_embedding(Deck().cards, 40)
    assert 40 == len(embeddings)


def test_padding():
    cards = [Card("cups", 1), Card("cups", 2)]
    result = cards_embedding(cards, 3)
    assert_array_equal(np.array([0, 1, 0]), result)


def test_deck_embedding():
    deck = Deck([Card("cups", 2)])
    expected = np.zeros(shape=(40,), dtype=int)
    expected[1] = 1
    assert_array_equal(expected, deck_embedding(deck))

    full_deck = Deck()
    assert_array_equal(np.ones(shape=(40,), dtype=int), deck_embedding(full_deck))
