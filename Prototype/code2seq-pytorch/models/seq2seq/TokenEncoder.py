from torch import nn, Tensor
import torch


def sequence_mask(self, lengths, maxlen=None, dtype=torch.bool):
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix

    mask.type(dtype)
    return mask


class TokenEncoder(nn.Module):
    def __init__(self, token_len, vocab_size, embedding_size, pad_idx=None):
        super(TokenEncoder, self).__init__()

        self.token_len = token_len
        self.num_embeddings = vocab_size
        self.embedding_size = embedding_size
        self.pad_idx = pad_idx

        if pad_idx is not None:
            self.embedding = nn.Embedding(
                vocab_size, embedding_size, padding_idx=pad_idx
            )
        else:
            self.embedding = nn.EmbeddingBag(vocab_size, embedding_size, mode="sum")

    def forward(self, contexts: Tensor):
        """
            Args:
                contexts: start tokens of the context. Must be of shape
                (batch_size, max_token_len)
                context_masks: (batch_size, )
                lengths: (batch_size, )
            Return:
                a tensor of shape (batch_size, embedding_size))
        """

        embedded = self.embedding(contexts)

        if self.pad_idx is None:
            return embedded

        return embedded.sum(dim=1)
