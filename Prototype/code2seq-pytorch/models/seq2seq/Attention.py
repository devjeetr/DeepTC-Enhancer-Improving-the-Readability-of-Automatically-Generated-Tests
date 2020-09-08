from torch import nn
from torch.nn import functional as F
import torch


class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim, device=None):
        super(Attention, self).__init__()

        self.attn = nn.Linear(
            encoder_hidden_dim + decoder_hidden_dim, decoder_hidden_dim
        )
        self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)

        if device:
            self.attn.to(device)
            self.v.to(device)

    def forward(self, hidden, context_encodings, context_masks=None):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [batch size, src len, , enc hid dim ]
        src_len = context_encodings.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        energy = torch.tanh(self.attn(torch.cat((hidden, context_encodings), dim=2)))
        attention = self.v(energy).squeeze(2)
        # attention = [batch size, src len]
        # make attention 0 for invalid contexts
        if context_masks is not None:
            # attention = attention.masked_fill(context_masks == 0, -1e10)
            # float 16
            attention = attention.masked_fill(context_masks == 0, -1e10)

        return F.softmax(attention, dim=1)
