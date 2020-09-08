from .Vocabulary import Vocabulary
from argparse import Namespace
from tokenizers import SentencePieceBPETokenizer
import pickle


class BPEVocabulary(Vocabulary):
    """ Represents a SentencePiece vocabulary for c2s.
    """

    def __init__(self, args: Namespace):
        super().__init__()

        self.target_encoder = SentencePieceBPETokenizer(
            args.target_vocab, args.target_merges
        )
        self.subtoken_encoder = SentencePieceBPETokenizer(
            args.subtoken_vocab, args.subtoken_merges
        )
        # self.target_encoder.add_special_tokens(
        #     [self.EOS_TOKEN, self.SOS_TOKEN, self.PAD_TOKEN]
        # )
        # self.subtoken_encoder.add_special_tokens([self.EOS_TOKEN, self.PAD_TOKEN])

        with open(args.node_dict, "rb") as f:
            self.node_to_index = pickle.load(f)
            self.index_to_node = {v: k for k, v in self.node_to_index.items()}

    def target_vocab_size(self):
        # print(self.target_encoder.num_special_tokens_to_add())
        return self.target_encoder.get_vocab_size() + 4

    def node_vocab_size(self):
        # print(self.target_encoder.num_special_tokens_to_add())
        return len(self.node_to_index) + 2

    def terminal_vocab_size(self):
        return self.subtoken_encoder.get_vocab_size() + 4

    def add_special_target_token(self, token: str):
        self.target_encoder.add_special_tokens([token])

    def add_special_terminal_token(self, token: str):
        self.subtoken_encoder.add_special_tokens([token])

    def encode_node(self, token_or_tokens):
        if isinstance(token_or_tokens, str):
            return self.node_to_index.get(
                token_or_tokens, self.node_to_index[self.UNK_TOKEN]
            )
        else:
            return list(map(self.encode_node, token_or_tokens))

    def decode_node(self, index_or_indices):
        if isinstance(index_or_indices, int):
            return self.index_to_node[index_or_indices]
        else:
            return list(map(self.decode_node, index_or_indices))

    def encode_target(self, token_or_tokens):
        if isinstance(token_or_tokens, str):
            return self.target_encoder.token_to_id(token_or_tokens)
        else:
            return self.target_encoder.encode(" ".join(token_or_tokens)).ids

    def decode_target(self, index_or_indices):
        if isinstance(index_or_indices, int):
            return self.target_encoder.id_to_token(index_or_indices)
        else:
            return self.target_encoder.decode(index_or_indices)

    def encode_terminal(self, token_or_tokens):
        if isinstance(token_or_tokens, str):
            return self.subtoken_encoder.token_to_id(token_or_tokens)
        else:
            return self.subtoken_encoder.encode(" ".join(token_or_tokens)).ids

    def decode_terminal(self, index_or_indices):
        if isinstance(index_or_indices, int):
            return self.terminal_encoder.id_to_token(index_or_indices)
        else:
            return self.terminal_encoder.decode(index_or_indices)
