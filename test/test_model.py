import unittest

# Imports from dlatent package
from .context import dlatent
from dlatent.latents import VectorQuant
from dlatent.encoders import LSTMEncoder
from dlatent.decoders import LSTMDecoder
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

		cls.vq3 = VectorQuant(embed_dim = cls.embed_dim,
						     num_embed = cls.num_embed, 
						     nd = cls.nd, 
						     commit_cost = cls.commit_cost,
						     device = device,
						     soft_train = True)

	def test_forward(self):

		enc_outputs = torch.randn(
			self.batchsize, self.hidden, self.seqlen, 
			device = torch.device(device)
		)
		self.vq1.sample(enc_outputs)
		self.vq1(enc_outputs)

		self.vq2.sample(enc_outputs)
		self.vq2(enc_outputs)

		self.vq3.sample(enc_outputs)
		self.vq3(enc_outputs)

	def test_backward(self):
		
		pass

class TestLSTMs(unittest.TestCase):
	# Tests LSTM encoder/decoders

	@classmethod
	def setUpClass(cls):

		cls.seqlen = 64
		cls.batchsize = 30

		cls.d_embedding = 200
		cls.d_hidden = 128
		cls.d_latent = 256
		cls.n_downsize = 2
		cls.kernel_size = 7

		cls.lstm_enc = LSTMEncoder(
			d_embedding = cls.d_embedding,
			d_hidden = cls.d_hidden,
			d_latent = cls.d_latent,
			n_downsize = cls.n_downsize,
			kernel_size = cls.kernel_size
		).to(torch.device(device))
		cls.lstm_dec = LSTMDecoder(
			d_embedding = cls.d_embedding,
			d_hidden = cls.d_hidden,
			d_latent = cls.d_latent,
			n_downsize = cls.n_downsize,
			kernel_size = cls.kernel_size
		).to(torch.device(device))

	def test_forwards(self):
		# Basically: just makes sure
		# forwards don't throw errors
		# and returns tensors of the right
		# shape.


		# Test encoder
		x = torch.randn(
			self.seqlen, self.batchsize, self.d_embedding, 
			device = torch.device(device)
		)
		out = self.lstm_enc(x)
		self.assertEqual(
			out.shape, torch.Size(
				[int(self.seqlen/(self.n_downsize**2)),
				 self.batchsize, 
				 self.d_latent]
			)
		)

		# Now test decoder
		out2 = self.lstm_dec(x, out)
		self.assertEqual(
			out2.shape, torch.Size(
				[self.seqlen - 1,
				 self.batchsize,
				 self.d_embedding
				]
			)
		)






if __name__ == '__main__':
	unittest.main()