import unittest
from .context import dlatent
from dlatent.latents import VectorQuant
import torch

# Use CPU because (a) more informative errors,
# (b) my computer is a toaster
device = 'cpu'

class TestVectorQuant(unittest.TestCase):

	@classmethod
	def setUpClass(cls):

		cls.embed_dim = 10
		cls.num_embed = 20
		cls.batchsize = 10
		cls.seqlen = 20
		cls.nd = 5
		cls.hidden = cls.embed_dim * cls.nd
		cls.commit_cost = 0.2

		cls.vq1 = VectorQuant(embed_dim = cls.embed_dim,
						     num_embed = cls.num_embed, 
						     nd = cls.nd, 
						     commit_cost = cls.commit_cost,
						     device = device,
						     gsoftmax = True)

		cls.vq2 = VectorQuant(embed_dim = cls.embed_dim,
						     num_embed = cls.num_embed, 
						     nd = cls.nd, 
						     commit_cost = cls.commit_cost,
						     device = device,
						     gsoftmax = False,
						     EMA = True)

	def test_vq(self):

		enc_outputs = torch.randn(self.batchsize, self.hidden, self.seqlen)
		self.vq1.sample(enc_outputs)
		self.vq1(enc_outputs)

		self.vq2.sample(enc_outputs)
		self.vq2(enc_outputs)




if __name__ == '__main__':
	unittest.main()