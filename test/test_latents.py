import unittest
from .context import dlatent
from dlatent.latents import VectorQuant

# Use CPU because (a) more informative errors,
# (b) my computer is a toaster
device = 'cpu'

class TestVectorQuant(unittest.TestCase):

	@classmethod
	def setUpClass(cls):

		embed_dim = 10
		num_embed = 20
		batchsize = 10
		seqlen = 20
		nd = 5
		commit_cost = 0.2

		cls.vq1 = VectorQuant(embed_dim = embed_dim,
						     num_embed = num_embed, 
						     nd = nd, 
						     commit_cost = commit_cost,
						     device = device,
						     gsoftmax = True)

	def test_vq(self):

		self.vq.sample()




if __name__ == '__main__':
	unittest.main()