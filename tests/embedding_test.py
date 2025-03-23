from lib.briscola.briscola import Deck 
from lib.briscola_env.embedding import card_embedding, card_reverse_embedding

def test_card_embeddings():
    for card in Deck().cards:
        embedding = card_embedding(card)  
        rev = card_reverse_embedding(embedding)
        assert card == rev
