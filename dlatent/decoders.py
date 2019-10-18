import torch
import torch.nn as nn

from .utils.weight_dropout import WeightDropout, dropout_dim

class LSTMDecoder(nn.Module):
    
    def __init__(self, d_embedding, d_hidden, d_latent, 
                    n_downsize=2, kernel_size=7,
                    weight_dropout=0.5, input_dropout=0.4, 
                    word_dropout=0.3):
        
        super().__init__()
        
        self.n_downsize = n_downsize
        self.weight_dropout = weight_dropout
        self.input_dropout = input_dropout
        self.word_dropout = word_dropout
        
        dims = [d_latent] + [d_hidden] * n_downsize
        conv_layers = [nn.Conv1d(d1, 2 * d2, kernel_size, 1, 
                            kernel_size // 2, groups=d1)
                            for d1, d2 in zip(dims[:-1], dims[1:])]
        self.conv_layers = nn.ModuleList(conv_layers)
        
        self.lstm = nn.LSTM(d_embedding + d_hidden, d_embedding)
        self.lstm = WeightDropout(self.lstm, 'weight_hh_l0', weight_dropout)
        
    # input: seqlen, batch, hidden
    def forward(self, x, z):
        """
        :param x: Actual text to reconstruct,
        seqlen by batch by hidden.
        :param z: (Possibly discretized) encoder outputs,
        seqlen/(n_downsize)**2 by batchsize by d_latent
        """
        
        x = dropout_dim(x, self.input_dropout, 0, self.training)
        x = dropout_dim(x, self.word_dropout, -1, self.training)
        
        for conv in self.conv_layers:
            z = conv(z.permute(1, 2, 0)).relu() # batch, hidden, seqlen
            z = z.transpose(1, 2)               # batch, seqlen, hidden
            z = z.reshape(z.size(0), 2 * z.size(1), -1)
            z = z.transpose(0, 1)               # seqlen, batch, hiddeno
            
        x = torch.cat((x[:-1], z[1:]), -1)
        x, _ = self.lstm(x)
        
        return x