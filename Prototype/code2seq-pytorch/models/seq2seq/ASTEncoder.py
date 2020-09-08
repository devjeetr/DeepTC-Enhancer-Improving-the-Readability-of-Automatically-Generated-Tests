from torch import nn
import torch


class ASTEncoder(nn.Module):
    def __init__(
        self,
        context_len: int,
        num_embeddings: int,
        embedding_size: int,
        hidden_size: int,
        bidirectional: bool = True,
        pad_idx=None,
        dropout=0.5,
    ):
        super(ASTEncoder, self).__init__()
        self.context_len = context_len
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        if pad_idx is not None:
            self.pad = True
            self.pad_idx = pad_idx
            self.embedding = nn.Embedding(
                num_embeddings, embedding_size, padding_idx=pad_idx
            )
        else:
            self.pad = False
            self.embedding = nn.Embedding(num_embeddings, embedding_size)

        # make sure batch_first=true
        self.lstm = nn.LSTM(
            embedding_size,
            hidden_size,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )

    def forward(self, contexts: torch.Tensor, context_lengths=None):
        """
            Args:
                contexts: A tensor containing the AST Path contexts of dimensions
                        (batch_size, context_len)
                context_lengths: A tensor of shape (batch_size) which
                                contains the length of each context, to be used when
                                we are packing the padded context vector
                hidden_state: The initial hidden state.

            Return:
                tensor containing forward and backward hidden states, of shape
                (batch_size, self.hidden_size * 2)
        """
        if self.pad and context_lengths is None:
            raise ValueError(
                "When pad_idx is provided to ASTEncoder, context lengths must also be provided to ASTEncoder.forward()"
            )

        embedded = self.embedding(contexts)

        hidden = None
        output = None
        if self.pad:
            embedded_packed = torch.nn.utils.rnn.pack_padded_sequence(
                embedded, context_lengths, batch_first=True, enforce_sorted=False,
            )
            output, (hidden, cell_state) = self.lstm(embedded_packed,)
        else:
            output, (hidden, cell_state) = self.lstm(embedded)

        return torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
