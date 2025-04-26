from lib.briscola.game import BriscolaDeck

def test_shuffle_seeding():
	for seed in range(1_000):
		a = BriscolaDeck()
		b = BriscolaDeck()
		a.shuffle(seed)
		b.shuffle(seed)
		for (l, r) in zip(a.cards, b.cards):
			assert l == r
