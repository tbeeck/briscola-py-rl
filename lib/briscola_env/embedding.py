import numpy as np
from typing import List
from lib.briscola.briscola import Deck
from lib.briscola.game import Game, Card


# Given a game and the current player, generate an embedding for the model
# Our hand (3)
# The trick so far (3)
# The Briscola (1)
# Cards still in the deck (40)
# our points (1)
# opponent points (3)
def game_embedding(game: Game, player: int):
    hand = cards_embedding(game.players[player].hand, 3)
    trick = cards_embedding(game.trick, 3)
    briscola = card_embedding(game.briscola)
    deck_cards = deck_embedding(game.deck)


def deck_embedding(deck : Deck):
    result = np.zeros(shape=(40, 1))

    

def cards_embedding(cards: List[Card], length: int):
    if len(cards) > length:
        raise Exception("cards exceeded expected length")
    result = np.zeros(shape=(length, 1))
    for i in range(len(cards)):
        result[i] = card_embedding(cards[i])
    return result


def card_embedding(card: Card) -> int:
    offset = Card.SUITS.index(card.suit) * 11
    return card.rank + offset

def card_reverse_embedding(i: int) -> Card:
    suit_index = 3
    while i > 11:
        i -= 11
        suit_index -= 1
    return Card(Card.SUITS[suit_index], i)
