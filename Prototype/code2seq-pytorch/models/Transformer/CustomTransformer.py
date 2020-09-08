import torch
from einops import rearrange
from torch import nn

from models.seq2seq.ContextLSTMEncoder import ContextLSTMEncoder


class CustomTransformer(nn.Module):
    def __init__(
        self,
        max_contexts: int = 100,
        subtoken_len: int = 7,
        subtoken_embedding_dims: int = 128,
        subtoken_vocab: int = -1,
        subtoken_pad_idx: int = -1,
        ast_path_len: int = 8,
        ast_embedding_dims: int = 128,
        ast_vocab: int = -1,
        ast_hidden_size: int = 256,
        ast_pad_idx: int = -1,
        label_vocab: int = -1,
        label_pad_idx: int = -1,
        bidirectional=True,
        context_embedding_dim: int = 512,
        context_encoder_dropout: float = 0.4,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        transformer_feedforward: int = 256,
        transformer_dropout: float = 0.4,
        transformer_activation: str = "relu",
    ):
        """[summary]

        Arguments:
            config {Config} -- The config object

        Keyword Arguments:
            d_model {int} -- [description] (default: {512})
            nhead {int} -- [description] (default: {8})
            num_encoder_layers {int} -- [description] (default: {3})
            num_decoder_layers {int} -- [description] (default: {3})
            transformer_feedforward {int} -- [description] (default: {256})
            transformer_dropout {float} -- [description] (default: {0.1})
            transformer_activation {str} -- [description] (default: {"relu"})
            device {[type]} -- [description] (default: {None})
        """

        super(CustomTransformer, self).__init__()

        self.path_encoder = ContextLSTMEncoder(
            max_contexts,
            subtoken_len,
            subtoken_embedding_dims,
            subtoken_vocab,
            ast_path_len,
            ast_vocab,
            ast_embedding_dims,
            ast_hidden_size,
            context_embedding_dim,
            ast_bidirectional=bidirectional,
            ast_pad_idx=ast_pad_idx,
            token_pad_idx=subtoken_pad_idx,
            dropout=context_encoder_dropout
        )

        self.transformer = torch.nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=transformer_feedforward,
            dropout=transformer_dropout,
            activation=transformer_activation,
        )

        self.d_model = d_model
        self.trg_pad_idx = label_pad_idx
        self.label_embedding = nn.Embedding(
            label_vocab, d_model, padding_idx=self.trg_pad_idx,
        )

        self.fc = nn.Linear(self.d_model, label_vocab)

    def make_subsequent_mask(self, trg):
        trg_len = trg.shape[1]

        return torch.triu(torch.ones((trg_len, trg_len)) * float("-inf"), diagonal=1).to(trg.device)

    def make_trg_pad_mask(self, trg):
        trg_pad_mask = (trg == self.trg_pad_idx).bool().to(trg.device)

        return trg_pad_mask

    def forward(
        self,
        start,
        end,
        path,
        start_lengths,
        end_lengths,
        path_lengths,
        context_mask,
        label,
    ):
        label_embed = self.label_embedding(label)
        label_embed = rearrange(label_embed, "b s d -> s b d")

        encoder_paths = self.path_encoder(
            start, end, path, start_lengths, end_lengths, path_lengths
        )
        encoder_paths = rearrange(encoder_paths, "b s d -> s b d")

        target_padding_mask = self.make_trg_pad_mask(label)
        target_subsequent_mask = self.make_subsequent_mask(label)

        context_mask = (
            context_mask.masked_fill(context_mask == 1.0, False)
            .masked_fill(context_mask == 0.0, True)
            .type(torch.bool)
        )

        decoded = self.transformer(
            encoder_paths,
            label_embed,
            tgt_mask=target_subsequent_mask,
            src_key_padding_mask=context_mask,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=context_mask.clone(),
        )

        decoded = rearrange(decoded, "s b d -> b s d")

        return self.fc(decoded)
