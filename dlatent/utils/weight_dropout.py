""" Creates WeightDropout class required for constructing the AWD-LSTM"""
import torch
import torch.nn as nn

def dropout_dim(input, p, dim, training):
    if not training:
        return input
    size = list(input.size())
    if dim is not None:
        size[dim] = 1
    mask = torch.empty(size, dtype=input.dtype, device=input.device) \
            .bernoulli_(1 - p) / (1 - p)
    return input * mask

class MultiplyInplace(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, other, out):
        ctx.save_for_backward(input, other)
        torch.mul(input, other, out=out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, other = ctx.saved_tensors
        return other * grad_output if ctx.needs_input_grad[0] else None, \
               input * grad_output if ctx.needs_input_grad[1] else None, \
               None

class WeightDropout(nn.Module):

    def __init__(self, module, name, p, dim=None):
        super(WeightDropout, self).__init__()
        self.module = module
        self.name = name
        weight = getattr(module, name)
        weight.requires_grad_(False)
        self.device = weight.device
        self.weight = nn.Parameter(weight.clone().detach().requires_grad_(True))
        self.p = p
        size = list(self.weight.size())
        if dim is not None:
            size[dim] = 1
        self.register_buffer('mask', torch.empty(size, dtype=self.weight.dtype))

    def forward(self, *args):
        if self.training:
            self.mask.bernoulli_(1 - self.p).div_(1 - self.p)
        else:
            self.mask.fill_(1)
        setattr(self.module, self.name, nn.Parameter(MultiplyInplace.apply(
                    self.weight, self.mask, getattr(self.module, self.name))))
        return self.module.forward(*args)