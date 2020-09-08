import random

import torch
from torch import nn

from models.seq2seq.Attention import Attention
from models.seq2seq.ContextLSTMEncoder import ContextLSTMEncoder
from models.seq2seq.Decoder import Decoder
from utilities.config import Config, ModelConfig


def encoder_from_config(config: Config, device: torch.device):
    encoder = ContextLSTMEncoder(
        config.general_params.max_contexts,
        config.general_params.subtoken_length,
        config.model_config.subtoken_embedding_size,
        config.model_config.subtoken_embedding_vocab_size,
        config.general_params.ast_path_length,
        config.model_config.ast_embedding_vocab_size,
        config.model_config.ast_embedding_size,
        config.model_config.ast_hidden_size,
        config.model_config.encoder_hidden_dim,
        ast_bidirectional=config.model_config.ast_bidirectional,
        ast_pad_idx=config.vocabularies.node_to_index[
            config.special_characters.node_pad_token
        ]
    )

    return encoder


def decoder_from_config(config: Config, device: torch.device):
    attention = Attention(
        config.model_config.encoder_hidden_dim, config.model_config.decoder_hidden_dim
    )
    decoder = Decoder(
        config.model_config.decoder_embedding_vocab_size,
        config.model_config.decoder_output_dim,
        config.model_config.encoder_hidden_dim,
        config.model_config.decoder_hidden_dim,
        config.model_config.decoder_dropout,
        attention,
        target_pad_idx=config.vocabularies.target_to_index[
            config.special_characters.target_pad_token
        ],
        device=device,
    )

    return decoder


class Seq2Seq(nn.Module):
    def __init__(self, config: Config, device: torch.device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder_from_config(config, device)
        self.decoder = decoder_from_config(config, device)
        self.config = config
        self.device = device

    def forward(
        self,
        start,
        end,
        path,
        start_len,
        end_len,
        path_len,
        context_mask,
        label,
        teacher_force=0.0,
    ):
        batch_size = start.shape[0]
        embedded = self.encoder(start, end, path, start_len, end_len, path_len)
        # divide by context length of each
        decoder_initial = embedded.sum(dim=1) / context_mask.sum(
            dim=1
        ).repeat_interleave(embedded.shape[2]).reshape(batch_size, -1)

        model_output = torch.zeros(
            self.config.general_params.target_length + 1,
            batch_size,
            self.config.model_config.decoder_embedding_vocab_size,
            device=self.device,
        )

        prediction = label[:, 0]
        hidden = decoder_initial

        for i in range(1, 2 + self.config.general_params.target_length):
            prediction, hidden = self.decoder(
                prediction, hidden, embedded, context_mask
            )

            model_output[i - 1] = prediction

            top1 = prediction.argmax(1)
            # top1 = prediction.max(1)[1]
            assert list(top1.shape) == list(label[:, i].shape)
            # prediction = label[:, i]
            use_teacher_force = random.random() < teacher_force

            prediction = label[:, i] if use_teacher_force else top1

        return model_output
