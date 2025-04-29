from lib.briscola.game import BriscolaCard, BriscolaDeck, BriscolaGame

def test_shuffle_seeding():
	for seed in range(1_000):
		a = BriscolaDeck()
		b = BriscolaDeck()
		a.shuffle(seed)
		b.shuffle(seed)
		for (l, r) in zip(a.cards, b.cards):
			assert l == r

def test_trick_scoring():
	game = BriscolaGame()
	game.trick = [BriscolaCard("batons", 7),BriscolaCard("batons", 1)]
	_, winning_card = game.trick_winner()
	assert BriscolaCard("batons", 1) == winning_card
