import torch
from .utils import gumbel_softmax

# Loosely based on https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
# which is based on the original:
# https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb
# from Oord et. al 2017

def calc_dists_and_latents(embeddings, 
                           enc_outputs,
                           compute_encodings = True):
    """Given embeddings and stacked encoder outputs, returns 
    the tensor latents as well as the discretized encodings
    param embeddings: nd x embed_dim x num_embed dimension tensor
    param enc_outputs: encoder outputs, stacked as
    (batchsize * seqlen) x nd x embed_dim. 
    Note hidden / nd must equal embed_dim """

    nd = embeddings.shape[0]

    # Calculate pairwise distances (sums/dot over embedding dimension, called 'h')
    # This means distances are (batchsize * seqlen) x nd x num_embed
    distances = enc_outputs.pow(2).sum(2).unsqueeze(-1) +\
                embeddings.pow(2).sum(1).unsqueeze(0) -\
                2*torch.einsum('dhk,bdh->bdk', [embeddings, enc_outputs])
    
    # Possibly just return the distances
    if not compute_encodings:
      return distances
    
    # Loop through nd - this is required since index_select is only 1 dimensional
    all_latents = []
    all_encodings = []
    for d in range(nd):
      
      # Else compute minimums
      dists, latents = distances[:, d, :].min(1)
      encodings = embeddings[d].index_select(dim = 1, index = latents)
      all_encodings.append(encodings)
      all_latents.append(latents)
    
    # Latents: (batchsize * seqlen) x nd
    latents = torch.stack(all_latents)
    # Encodings: embed_dim x (batchsize * seqlen) x nd
    encodings = torch.stack(all_encodings)
    return encodings, latents, distances

class VectorQuant(torch.nn.Module):
  
  def __init__(self, embed_dim, num_embed, nd, commit_cost,
               device, soft_train = False,
               gsoftmax = False,
               EMA = False, decay = 0.99, epsilon = 1):
    """
    :param embed_dim: dimensionality of embeddings, e.g. 150
    :param num_embed: number of embeddings per latent
    :param nd: the discretization constant (in DVQ-VAE), i.e. the number of separate
    "chunks" to run kmeans on. When nd = 1, this is a vanilla VQ-VAE.
    :param commit_cost: hparam (beta) from paper
    :param bool EMA: if true, use exponentially-decaying moving averages
    :param bool soft_train: if true, train using KL loss
    :param bool gsoftmax: if true, don't use VQ-VAE but use gumbel-softmax sampling"""
    
    super(VectorQuant, self).__init__()
    
    # save hparams
    self.num_embed = num_embed
    self.embed_dim = embed_dim 
    self.commit_cost = commit_cost # Called beta in oord et al.
    self.device = torch.device(device)
    self.soft_train = soft_train
    self.gsoftmax = gsoftmax
    self.nd = nd
    self.EMA = EMA

    # Some conflicts
    if self.EMA and self.gsoftmax:
      raise ValueError('Cannot use both gsoftmax and EMA training')
          
    # Initialize embeddings and (possibly) EMA cache
    if self.EMA:
      self.decay = decay
      self.epsilon = epsilon
      self.register_buffer('ema_counts', torch.zeros(nd, num_embed, 
                                                     device = self.device,
                                                     requires_grad = False) + 10)
      # When using EMA, we don't need (or want) any gradients on the latent embeddings
      self.register_buffer(
        'embeddings', torch.randn(nd, embed_dim, num_embed, 
                                  device = self.device, 
                                  requires_grad = False)
        )
      self.ema_sums = 5 * self.embeddings
    else:
      # Otherwise the embeddings are a parameter
      self.embeddings = torch.randn(
        nd, embed_dim, num_embed, device = self.device,
         requires_grad = True
      )

      
    # If we're soft training, use a uniform prior
    if self.soft_train:
      self.register_buffer('prior', torch.ones(self.num_embed)/self.num_embed)
      self.prior = self.prior.detach().to(self.device)
    
    
  def stack_enc_outputs(self, enc_outputs):
    """
    param enc_outputs: encoder outputs, stacked as a 
    batchsize x hidden x seqlen dimension tensor.
    returns: (enc_outputs, batchsize, seqlen), where:
      1. enc_outputs are stacked as a
      (batchsize * seqlen) x nd x embed_dim dimension tensor.
      2. batchsize, seqlen were the initial parameters
    """
    # Get parameters
    batchsize, seqlen = enc_outputs.size(0), enc_outputs.size(2)

    # Re-view
    enc_outputs = enc_outputs.view(batchsize*seqlen, self.nd, -1)

    return enc_outputs, (batchsize, seqlen)
  
  def split_outputs(self, encodings, batchsize, seqlen):

    # Undoes "stack_enc_outputs, but is usually applied to
    # discretized encodings
    encodings = encodings.contiguous().view(batchsize, -1, seqlen)
    return encodings
    
  def forward(self, inputs,  
              temperature = 1,
              return_latents = False):
    """
    :param inputs: outputs of the encoder,  stacked as a 
    batchsize x hidden x seqlen dimension tensor.
    :param temperature: temperature for gumbel-softmax sample, if the class
    was initialized with gsample = True.
    :param bool return_latents: if True, just return the latents (not their embeddings)'
    Returns:
        Outputs: the quantized vector. 
        Loss: the loss to add to backprop through.
        Commit_loss: the distance between the cts vectors and the quantized output
    """
      
    # Reshape encoder outputs
    enc_outputs, (batchsize, seqlen) = self.stack_enc_outputs(inputs)
 
    # Possibly get the latents via gumbel-softmax sampling
    if self.gsoftmax and not return_latents:
      encodings, latents, distances = self.gs_sample(
        enc_outputs, temperature = temperature
      )
    # Else, nearest-neighbor search
    else:
      encodings, latents, distances = calc_dists_and_latents(self.embeddings, enc_outputs)

    # Compute KL Loss 
    if self.soft_train and self.training:
      distprobs = distances / distances.sum(2).unsqueeze(2) # Create distribution over latents
      kl = torch.einsum('bdk,bdk->b', [torch.log(distprobs/self.prior), distprobs]).mean()

    # Helpful for analyzing index collapse and running kmeans, return latents
    if return_latents:
      return latents
    
    # Possible EMA update -------------------------------------
    if self.EMA and self.training:

      # Get new counts and new sums
      new_counts = []
      new_sums = [
        torch.zeros(
          self.embed_dim, self.num_embed, requires_grad = False
          ).to(self.device) for i in range(self.nd)
        ]

      for d in range(self.nd):
        nd_latent = latents[d, :]
        new_sums[d].index_add_(1, 
                               nd_latent,
                               enc_outputs[:, d].transpose(0, 1))
        new_counts.append(
            torch.bincount(nd_latent, minlength = self.num_embed)
        )

      # Make them all floats
      # ema_counts, new_couts are nd x num_embed
      # ema_sums, new_sums are nd x embed_dim x num_embed
      self.ema_counts = self.ema_counts.float()
      new_sums = torch.stack(new_sums).float()
      new_counts = torch.stack(new_counts).float()


      # Exponential moving average updates - epsilon prevents div0 errors
      self.ema_counts = self.decay * self.ema_counts + (1 - self.decay) * new_counts
      self.ema_sums = self.decay * self.ema_sums + (1 - self.decay) * new_sums
      new_embed = self.ema_sums.transpose(1, 2) / self.ema_counts.unsqueeze(-1)
      
      # Update embeddings
      self.embeddings = new_embed
    
    # Resplit and create output using straight-through gradient. -----------
    encodings = self.split_outputs(
      encodings, batchsize = batchsize, seqlen = seqlen
    )
    if not self.gsoftmax: 
      output = inputs + encodings.detach() - inputs.detach()
    # Of course, if we're using gsoftmax sampling we don't need a straight-through
    else:
      output = encodings
    
    # Pass some gradient to embeddings/create commit loss
    commit_loss = (inputs - encodings.detach()).pow(2).mean()

    if self.gsoftmax:
      loss =  self.commit_cost * commit_loss
      distance_loss = (encodings - inputs.detach()).pow(2).mean()
      loss = self.commit_cost * commit_loss + distance_loss
    else:
      loss = self.commit_cost * commit_loss
    if self.soft_train and self.training:
      loss += kl
    else:
      kl = torch.tensor(0)   
    return output, loss, commit_loss, kl
  
  
  def gs_sample(self, 
                inputs_stacked, 
                temperature = 1,
                **kwargs):
        
    # This means distances are (batchsize * seqlen) x nd x num_embed
    distances = calc_dists_and_latents(self.embeddings, 
                                       inputs_stacked, 
                                       compute_encodings = False)
    
    # Latents are same dimensionality
    latents = gumbel_softmax.gumbel_softmax(
      dim = 2,
      logits = distances,
      temperature = temperature,
      device = self.device
    )

    # Encodings are (batchsize*seqlen) x nd x hidden
    encodings = torch.einsum('dhk,bdk->bdh', [self.embeddings, latents])
    return encodings, latents, distances
    

  def sample(self, inputs, K = 1,
             bit_flips = True,
             bit_flip_prob = 0.03,
             split_latents = True):
    """ Samples latents (not using gumbel-softmax) two methods.
    :param inputs: encoder outputs
    :param K: number of samples to take 
    :param bit_flips: If true, resample each coordinate of the
    argmax latents with some probability. If false, sample every
    single coordinate of the latents.
    :param bit_flip_prob: When bit_flips is true, the probability
    of "flipping" each coordinate.
    :param split_latents: If true, reshape latents so that
    batchsize and seqlen are different coordinates.
    returns: inds (one-hot latents), log_probs (log probability of each coord)
    """
    
    inputs_stacked, (batchsize, seqlen) = self.stack_enc_outputs(inputs)
       
    # Naive O(nm) nearest neighbor search 
    distances = calc_dists_and_latents(self.embeddings, inputs_stacked,
                                       compute_encodings = False) 
    latent_probs = distances / distances.sum(2).unsqueeze(2)
    # latent_probs dim: (batchsize*seqlen x nd x num_embed)
    latent_probs = latent_probs.clamp(min = 1e-6, max = 1 - 1e-6).log() # Watch out for infs/nans

    # Helpers
    log_probs = []
    inds = []
    L = latent_probs.size(0)
    helper_indexer = torch.stack([torch.arange(0, L, 1) for i in range(K)])

    # Sample and get log probs for each nd
    for d in range(self.nd):
      
      # Sample
      latent_probs_nd = latent_probs[:, d]
      inds_nd = torch.distributions.Categorical(
          logits = latent_probs_nd
      ).sample(torch.Size([K]))
      inds.append(inds_nd)

      # Index into log probabilities
      log_probs_nd = latent_probs_nd[helper_indexer, inds_nd]
      log_probs.append(log_probs_nd)    

    
    # Reshape: nd x numsample (K) x (batchsize * seqlen)
    inds = torch.stack(inds)
    log_probs = torch.stack(log_probs)
    
    # Possibly use the second sampling method (basically, 
    # we construct a binary mask and when the binary mask 
    # is False, we use the argmax latent)
    if bit_flips:
      _, latents, _ = calc_dists_and_latents(self.embeddings, inputs_stacked,
                                       compute_encodings = True) 
      latents = latents.view(self.nd, 1, inds.shape[2])
      mask = torch.distributions.Bernoulli(bit_flip_prob).sample(
        torch.Size([self.nd, K, inds.shape[2]]), 
      ).long().to(self.device)
      inds = mask * inds + (1 - mask) * latents

    # Possibly split to nd x numsample x batchsize x seqlen
    if split_latents:
      log_probs = log_probs.view(self.nd, K, batchsize, seqlen)
      inds = inds.view(self.nd, K, batchsize, seqlen)
      
    if bit_flips:
      return inds, None
    
    return inds, log_probs

  def reset_embeddings(self, embeddings):
    
    if set(embeddings.shape) != set(self.embeddings.shape):
      raise ValueError(f"New embeddings shape {embeddings.shape} disagrees with old embeddings shape {self.embeddings.shape}")

    inplace_add(self.embeddings, embeddings - self.embeddings)
    assert self.embeddings == embeddings
    return None
