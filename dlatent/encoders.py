import torch
import torch.nn as nn

from .utils.weight_dropout import WeightDropout, dropout_dim


class LSTMEncoder(nn.Module):
    """
    LSTM encoder.
    :param d_embedding: dimension of WORD embeddings
    :param d_hidden: dimension of hidden state 
    :param d_latent: dimension of LATENT embeddings
    :param n_downsize: Number of times to downsize by
    a factor of 2.
    :param kernel_size: Kernel to use in downsizing step.
    """
    
    def __init__(self, d_embedding, d_hidden, d_latent,
                    n_downsize=2, kernel_size=7,
                    weight_dropout=0.5, input_dropout=0.4, 
                    inter_dropout=0.3, output_dropout=0.4):
        
        super().__init__()
        
        # Save parameters
        self.n_downsize = n_downsize
        self.weight_dropout = weight_dropout
        self.input_dropout = input_dropout
        self.inter_dropout = inter_dropout
        self.output_dropout = output_dropout
        
        # LSTM layers with weightdropout
        dims = [d_embedding] + [d_hidden] * (n_downsize + 1)
        lstm_layers = [nn.LSTM(d1, d2 // 2, bidirectional=True) 
                           for d1, d2 in zip(dims[:-1], dims[1:])]
        lstm_layers = [WeightDropout(l, 'weight_hh_l0', weight_dropout) 
                           for l in lstm_layers]
        self.lstm_layers = nn.ModuleList(lstm_layers)
        

        # Downsizing convolutional layers
        conv_layers = [nn.Conv1d(d_hidden, d_hidden, kernel_size, 2, 
                            kernel_size // 2, groups=d_hidden) 
                            for _ in range(n_downsize)]
        self.conv_layers = nn.ModuleList(conv_layers)
        
        # Final output
        self.linear = nn.Linear(d_hidden, d_latent)
        
    def forward(self, x):
        """
        :param x: torch tensor of size seqlen by batchsize by d_embedding

        returns: output, torch tensor of size seqlen/(n_downsize)**2 by
        batchsize by d_latent
        """
        
        x = dropout_dim(x, self.input_dropout, 0, self.training)
        
        for lstm, conv in zip(self.lstm_layers, self.conv_layers):
            x, _ = lstm(x)
            x = dropout_dim(x, self.inter_dropout, 0, self.training)
            x = conv(x.permute(1, 2, 0)).permute(2, 0, 1).relu()
        x, _ = self.lstm_layers[-1](x)
        x = dropout_dim(x, self.output_dropout, 0, self.training)
        
        x = self.linear(x)
        
        return x                
