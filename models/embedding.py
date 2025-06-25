import torch
import torch.nn as nn
import math
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer

class EmbeddingLayer(nn.Module):
  def __init__(self, dim, vocab_dim):
    super().__init__()
    self.embedding = nn.Parameter(
      torch.empty((vocab_dim, dim))
    )
    torch.nn.init.kaiming_uniform_(
      self.embedding, a=math.sqrt(5)
    )

  def forward(self, x):
    return self.embedding[x]

class Embedding(torch.nn.Module):

    def __init__(self, hidden_size, vocab_size):
        super(Embedding, self).__init__()
        self.p_enc_1d_model_sum = Summer(PositionalEncoding1D(hidden_size))
        self.vocab_embed = EmbeddingLayer(
            hidden_size, vocab_size
        )
    
    def forward(self, x, dummy_sigma):
        # x shape: [batch_size, seq_len, hidden_size]
        # dummy_sigma shape: [batch_size, 1]
        # Apply positional encoding
        x = self.vocab_embed(x)
        x = self.p_enc_1d_model_sum(x)
        
        return x