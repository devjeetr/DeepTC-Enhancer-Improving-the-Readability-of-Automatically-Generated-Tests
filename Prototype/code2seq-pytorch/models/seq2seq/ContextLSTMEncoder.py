from torch import nn
import torch

from .ASTEncoder import ASTEncoder
from .TokenEncoder import TokenEncoder


class ContextLSTMEncoder(nn.Module):
    def __init__(
        self,
        max_contexts: int,
        token_len: int,
        token_embedding_size: int,
        token_num_embeddings: int,
        ast_path_len: int,
        ast_num_embeddings: int,
        ast_embedding_size: int,
        ast_hidden_size: int,
        combined_hidden: int,
        token_pad_idx: int = None,
        ast_bidirectional: bool = True,
        ast_pad_idx: int = None,
        dropout: float = 0.25,
    ):
        super(ContextLSTMEncoder, self).__init__()

        self.max_contexts = max_contexts
        self.token_len = token_len
        self.token_embedding_size = token_embedding_size
        self.token_num_embeddings = token_num_embeddings
        self.ast_path_len = ast_path_len
        self.ast_num_embeddings = ast_num_embeddings
        self.ast_embedding_size = ast_embedding_size
        self.ast_hidden_size = ast_hidden_size
        self.ast_bidirectional = ast_bidirectional
        self.combined_hidden = combined_hidden
        self.ast_pad_idx = ast_pad_idx

        # initialize layers
        self.ast_encoder = ASTEncoder(
            self.ast_path_len,
            self.ast_num_embeddings,
            self.ast_embedding_size,
            self.ast_hidden_size,
            self.ast_bidirectional,
            self.ast_pad_idx,
        )

        self.token_encoder = TokenEncoder(
            self.token_len, self.token_num_embeddings, self.token_embedding_size, pad_idx=token_pad_idx
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(
            2 * self.ast_hidden_size + 2 * self.token_embedding_size,
            self.combined_hidden,
        )

        self.out = nn.Tanh()

    def forward(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        path: torch.Tensor,
        start_len: torch.Tensor,
        end_len: torch.Tensor,
        ast_path_lens: torch.Tensor = None,
    ) -> torch.Tensor:
        """[summary]

        Arguments:
            start {torch.Tensor} -- start token of shape (bsz, seq_len)
            end {torch.Tensor} -- end token of shape (bsz, seq_len)
            path {torch.Tensor} -- ast path (basz, )
            start_len {torch.Tensor} -- [description]
            end_len {torch.Tensor} -- [description]

        Keyword Arguments:
            ast_path_lens {torch.Tensor} -- [description] (default: {None})

        Returns:
            torch.Tensor -- [description]
        """
        dtype = torch.float32

        batch_size = start.shape[0]
        # create context mask
        context_mask = ast_path_lens.clamp(0, 1).flatten()
        assert list(context_mask.shape) == [batch_size * self.max_contexts]

        # prepare inputs
        start, end, path = [
            t.reshape(batch_size * self.max_contexts, -1) for t in (start, end, path)
        ]

        assert list(start.shape) == [batch_size * self.max_contexts, self.token_len]
        assert list(end.shape) == [batch_size * self.max_contexts, self.token_len]
        assert list(path.shape) == [batch_size * self.max_contexts, self.ast_path_len]

        # non-zero context lengths
        ast_path_lens = ast_path_lens[ast_path_lens > 0]
        # compute masked embeddings
        start_embedded = self.token_encoder(start[context_mask == 1, :])
        end_embedded = self.token_encoder(end[context_mask == 1, :])
        path_embedded = self.ast_encoder(
            path[context_mask == 1, :], context_lengths=ast_path_lens
        )

        # unmask
        start_unmasked = torch.zeros(
            batch_size * self.max_contexts,
            self.token_embedding_size,
        ).type_as(start_embedded)
        start_unmasked[context_mask == 1] = start_embedded

        end_unmasked = torch.zeros(
            batch_size * self.max_contexts,
            self.token_embedding_size,
        ).type_as(end_embedded)
        end_unmasked[context_mask == 1] = end_embedded

        path_unmasked = torch.zeros(
            batch_size * self.max_contexts,
            2 * self.ast_hidden_size,
        ).type_as(path_embedded)
        path_unmasked[context_mask == 1] = path_embedded

        # concatenate
        combined = torch.cat([start_unmasked, path_unmasked, end_unmasked], dim=1)
        combined = self.dropout(combined)
        combined = combined.reshape(batch_size, self.max_contexts, -1)
        assert list(combined.shape) == [
            batch_size,
            self.max_contexts,
            2 * self.ast_hidden_size + 2 * self.token_embedding_size,
        ]

        combined = self.fc(combined)

        return self.out(combined)
