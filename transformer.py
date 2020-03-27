######################################################################
# Multi-task transformer for disfluency detection
#

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):

    def __init__(self, ntoken, nclstoken, ntagtoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout, activation="gelu")
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.cls_decoder = nn.Linear(ninp, nclstoken)
        self.tag_decoder = nn.Linear(ninp, ntagtoken)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.tag_decoder.bias.data.zero_()
        self.tag_decoder.weight.data.uniform_(-initrange, initrange)
        self.cls_decoder.bias.data.zero_()
        self.cls_decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src_key_padding_mask = src == 0
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        src = src.transpose(0,1)
        output = self.transformer_encoder(src, src_key_padding_mask = src_key_padding_mask)
        output = output.transpose(0,1)
        tag_output = self.tag_decoder(output[:,1:,:])
        cls_output = self.cls_decoder(output[:,0,:])
        return tag_output, cls_output


######################################################################
# ``PositionalEncoding`` module injects some information about the
# relative or absolute position of the tokens in the sequence. The
# positional encodings have the same dimension as the embeddings so that
# the two can be summed. Here, we use ``sine`` and ``cosine`` functions of
# different frequencies.
#

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


