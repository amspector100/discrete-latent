import torch

""" Sample gumbel softmax following https://arxiv.org/pdf/1611.01144.pdf 
Implementation is similar to https://github.com/dev4488/VAE_gumble_softmax"""

def sample_gumbel(shape, device, eps = 1e-20):
  device = torch.device(device)
  U = torch.Tensor(*shape).uniform_(0,1).to(device)
  return -1*(torch.log(-1*torch.log(U + eps) + eps))

def gumbel_softmax_sample(dim, logits, device, temperature, 
                          norm = 'softmax',
                          eps = 1e-20, ):
    y = logits + sample_gumbel(logits.shape,
                               eps = eps,
                               device = device)

    # Can possibly use the fact that distances are norms to normalize
    if norm == 'l2':
      if y.min() <= 0:
        raise ValueError('Cannot use l2 norm since some logits are negative')
      return y/y.sum(dim).unsqueeze(dim)
    # Alternatively, use softmadx
    else:
      return (y/temperature).softmax(dim)

def gumbel_softmax(dim, logits, temperature, device, 
                   norm = 'softmax', hard = False, eps = 1e-20):
  """ If hard == true, use a hard sample but with a straight through gradient
  (accomplished by the detach call)"""
  y = gumbel_softmax_sample(
    dim, logits, temperature, norm = norm, eps = eps, device = device
  )
  if hard:
    _, max_idx = y.max(dim)
    max_idx = max_idx.long().unsqueeze(dim)
    one_hot = torch.zeros(*y.shape)
    one_hot.scatter_(dim, max_idx, 1)
    y = (one_hot - y).detach() + y
  return y