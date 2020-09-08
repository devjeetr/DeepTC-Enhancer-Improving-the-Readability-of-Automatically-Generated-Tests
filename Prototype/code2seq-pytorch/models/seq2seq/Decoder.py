from torch import nn
import torch


class Decoder(nn.Module):
    def __init__(
        self,
        target_vocab_size,
        emb_dim,
        enc_hid_dim,
        dec_hid_dim,
        dropout,
        attention,
        target_pad_idx=None,
        device: torch.device = None,
    ):

        super(Decoder, self).__init__()

        self.target_vocab_size = target_vocab_size
        self.attention = attention

        if target_pad_idx is not None:
            print("setting target pad idx")
            self.embedding = nn.Embedding(
                target_vocab_size, emb_dim, padding_idx=target_pad_idx
            )
        else:
            self.embedding = nn.Embedding(target_vocab_size, emb_dim)

        self.rnn = nn.GRU(enc_hid_dim + emb_dim, dec_hid_dim)
        # self.rnn = nn.LSTM(enc_hid_dim + emb_dim, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(enc_hid_dim + dec_hid_dim + emb_dim, target_vocab_size)

        self.softmax = nn.LogSoftmax(dim=1)

        if device:
            self.embedding.to(device)
            self.rnn.to(device)
            self.fc_out.to(device)
            self.dropout.to(device)

    def forward(self, input, decoder_state, encoder_outputs, context_masks=None):
        # hidden, cell_context = decoder_state
        # input = [batch size] the previous prediction or ground truth if using teacher forcing
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim]

        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]
        a = self.attention(decoder_state, encoder_outputs, context_masks)
        # a = [batch size, src len]
        a = a.unsqueeze(1)

        # a = [batch size, 1, src len]

        weighted = torch.bmm(a, encoder_outputs)

        # weighted = [batch size, 1, enc hid dim * 2]
        weighted = weighted.permute(1, 0, 2)
        # weighted = [1, batch size, enc hid dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim=2)
        # rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]

        # output, hidden = self.rnn(rnn_input, (hidden.unsqueeze(0), cell_context.unsqueeze(0)))

        output, hidden = self.rnn(rnn_input, decoder_state.unsqueeze(0))

        # output = [seq len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden

        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))

        # prediction = [batch size, output dim]

        return prediction, hidden.squeeze(0)
