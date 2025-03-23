import numpy as np
from numpy.testing import assert_array_equal

from lib.briscola.briscola import Card, Deck 
from lib.briscola_env.embedding import cards_embedding, card_embedding, card_reverse_embedding

def test_card_embeddings():
    for card in Deck().cards:
        embedding = card_embedding(card)  
        rev = card_reverse_embedding(embedding)
        assert card == rev

def test_unique_embeddings():
    embeddings = cards_embedding(Deck().cards, 40)
    assert 40 == len(embeddings)

def test_padding():
    cards = [Card("cups", 1), Card("cups", 2)]
    result = cards_embedding(cards, 3)
    assert_array_equal(np.array([1, 2, 0]), result)
