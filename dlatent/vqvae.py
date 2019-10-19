import torch
import torch.nn as nn

from .encoders import LSTMEncoder
from .decoders import LSTMDecoder
from .latents import VectorQuant


class VQVAE(nn.Module):
	"""
	Wrapper which combines the embedding, encoder,
	decoder, and bottleneck (vectorquant) modules.
	:param embedding: word embedding module
	:param encoder: encoder module (i.e. LSTMEncoder)
	:param decoder: decoder module (i.e. LSTMDecoder)
	:param bottleneck: discretization bottleneck,
	e.g. VectorQuant class.
	"""
	
	def __init__(self, embedding, encoder, decoder, bottleneck):
		
		super().__init__()
		
		# Get parameters, save modules
		self.embedding = embedding
		self.embedding_dim = embedding.embedding_dim
		self.vocab_size = embedding.num_embeddings
		self.encoder = encoder
		self.decoder = decoder
		self.bottleneck = bottleneck
		
		# Final linear output layer
		self.linear = nn.Linear(self.embedding_dim, self.vocab_size,
								bias=False)
		self.linear.weight = self.embedding.weight
		
	def forward(self, x, discrete=True, **kwargs):
		"""
		:param x: one-hot inputs, seqlen x batchsize
		:param bool discrete: If true, apply the 
		bottleneck/vectorquant class
		:param **kwargs: kwargs to the bottleneck forward 
		function.
		"""

		# Embedding and encoder
		x = self.embedding(x)
		z = self.encoder(x)

		# Reshape for bottleneck: from 
		# seqlen x batchsize x d_hidden to
		# batchsize x d_hidden x seqlen
		z = z.transpose(0, 1).transpose(1, 2)

		# Bottleneck
		if discrete:
			z, vq_loss = self.bottleneck(z, **kwargs)
		else:
			vq_loss = (0, 0, 0)
		
		# Reshape for decoder: 
		# from batchsize x d_hidden x seqlen
		# to seqlen x batchsize x d_hidden
		z = z.transpose(1, 2).transpose(0, 1)

		# Decoder 
		x = self.decoder(x, z)
		x = self.linear(x)
		return x, vq_loss

	@classmethod
	def from_params(cls,
					vocab_size,
					padding_idx,
					d_embedding,
					d_hidden,
					d_latent,
					num_latent_embed,
					device = 'cuda',
					nd = 1,
					n_downsize = 1,
					kernel_size = 7, 
					weight_dropout = 0.5,
					input_dropout = 0.4,
					inter_dropout = 0.4,
					output_dropout = 0.4,
					word_dropout = 0.3,
					**kwargs
					):
		""" Wrapper function which constructs
		a VQVAE using the given parameters.
		:param vocab_size: Number of words in your vocabulary.
		:param padding_idx: Which integer corresponds to the 'pad' word.
		:param d_embedding: Dimensionality of WORD embeddings.
		:param d_hidden: Dimensionality of hidden states of LSTM encoder/decoders.
		:param d_latent: Dimensionality of LATENT embeddings.
		:param num_latent_embed: Number of latent embeddings.
		:param device: defaults to 'cuda', the device to run the model on.
		:param nd: Defaults to 1, the number of chunks to 
		take maxes across in the discretization step (e.g. the 'D' in 'DVQ').
		('nd' stands for num_decompose).    
		:param kernel_size: Kernel size of convolutional down/upsizing layers.
		:param **kwargs: Other arguments to the VectorQuant class.
		These are VERY IMPORTANT - see the VectorQuant class docs for their
		default behavior.
		"""

		# Construct pieces
		embedding = nn.Embedding(vocab_size, d_embedding, padding_idx=padding_idx)
		embedding = embedding.to(torch.device(device))
		encoder = LSTMEncoder(d_embedding, d_hidden, d_latent, n_downsize,
						  kernel_size = kernel_size, weight_dropout=weight_dropout,
						  input_dropout=input_dropout, inter_dropout=inter_dropout, 
						  output_dropout=output_dropout)
		encoder = encoder.to(torch.device(device))
		decoder = LSTMDecoder(d_embedding, d_hidden, d_latent, n_downsize, 
						  kernel_size = kernel_size, weight_dropout=weight_dropout,
						  input_dropout=input_dropout, word_dropout=word_dropout)
		decoder = decoder.to(torch.device(device))
		bottleneck = VectorQuant(
			embed_dim = int(d_latent / nd), num_embed = num_latent_embed,
			nd = nd, device = device, **kwargs
		)

		vqvae = cls(embedding, encoder, decoder, bottleneck)
		return vqvae