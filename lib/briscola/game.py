from typing import List, Tuple, Optional

from lib.briscola.briscola import BriscolaDeck, BriscolaCard, BriscolaPlayer

HAND_SIZE = 3


class BriscolaGame:
    def __init__(self, players=4, goes_first=0, seed=None):
        if players not in [2, 4]:
            raise ValueError(f"Invalid number of players: {players}")
        if goes_first < 0 or goes_first >= players:
            raise ValueError(f"Invalid first player index: {goes_first}")

        deck = BriscolaDeck()
        deck.shuffle(seed)
        briscola = deck.take(1)[0]

        self.deck = deck
        self.players = [BriscolaPlayer() for _ in range(players)]
        self.briscola = briscola
        self.trick = []
        self.action_on = goes_first
        self.deal_cards(HAND_SIZE)

    def play(self, card: BriscolaCard) -> None:
        if len(self.trick) == len(self.players):
            raise Exception("trick over")

        if isinstance(card, int):
            card = self.players[self.action_on].hand[card]

        if not self.playable(card):
            raise Exception(f"card not playable: {card}")

        self.players[self.action_on].remove_from_hand(card)
        self.trick.append(card)
        self.action_on = (self.action_on + 1) % len(self.players)

    def playable(self, card: Optional[BriscolaCard]) -> bool:
        if self.should_score_trick():
            return False
        if self.needs_redeal():
            return False
        if card is None or card not in self.players[self.action_on].hand:
            return False
        return True

    def score_trick(self) -> int:
        if len(self.trick) != len(self.players):
            raise Exception("trick not over")

        winning_player, _ = self.trick_winner()
        self.players[winning_player].take_trick(self.trick)
        self.trick = []
        self.action_on = winning_player

        return winning_player

    def trick_winner(self) -> Tuple[int, BriscolaCard]:
        trump = self.trump_suit()
        lead = self.lead_suit()

        winning_card = None
        for card in self.trick:
            if not winning_card:
                winning_card = card
            elif card.suit == trump and winning_card.suit != trump:
                winning_card = card
            elif card.suit == lead and winning_card.suit not in [trump, lead]:
                winning_card = card
            elif (
                card.suit == trump
                and winning_card.suit == trump
                and BriscolaCard.strength(card) > BriscolaCard.strength(winning_card)
            ):
                winning_card = card
            elif (
                card.suit == lead
                and winning_card.suit == lead
                and BriscolaCard.strength(card) > BriscolaCard.strength(winning_card)
            ):
                winning_card = card
        if not winning_card:
            raise Exception("no winning card")

        winning_player_index = (
            self.action_on + self.trick[::-1].index(winning_card)
        ) % len(self.players)
        return winning_player_index, winning_card

    def redeal(self):
        if all(len(p.hand) < 3 for p in self.players):
            self.deal_cards(1)
        else:
            raise Exception("players have cards")

    def deal_cards(self, n):
        cards = self.deck.take(n * len(self.players))
        split_cards = [cards[i :: len(self.players)] for i in range(len(self.players))]
        for i, player in enumerate(self.players):
            player.hand.extend(split_cards[i])

    def needs_redeal(self) -> bool:
        return (
            len(self.trick) == 0
            and len(self.deck.cards) > 0
            and all(len(p.hand) < HAND_SIZE for p in self.players)
        )

    def trump_suit(self) -> str:
        return self.briscola.suit

    def lead_suit(self) -> Optional[str]:
        return self.trick[-1].suit if self.trick else None

    def should_score_trick(self) -> bool:
        return len(self.trick) == len(self.players)

    def game_over(self) -> bool:
        return len(self.deck.cards) == 0 and all(len(p.hand) == 0 for p in self.players)

    def leaders(self) -> List[BriscolaPlayer]:
        return sorted(self.players, key=lambda p: p.score(), reverse=True)

    def __repr__(self):
        result = []
        for i in range(len(self.players)):
            result.append(f"Player {i}: {self.players[i]}")
        result.append(f"Action on: {self.action_on}")
        result.append(f"Trick: {self.trick}")
        return "\n".join(result)
