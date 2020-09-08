from .Vocabulary import Vocabulary
from argparse import Namespace
import pickle


def index_mapping_from_dict(dict, max_size=-1, additional_keys={}):
    mapping, inverse = {}, {}

    counter = 0

    for key in additional_keys:
        mapping[key] = counter
        inverse[counter] = key
        counter += 1

    for key in dict:
        mapping[key] = counter
        inverse[counter] = key
        counter += 1

        if max_size != -1 and counter == max_size:
            break

    return mapping, inverse


class WordVocabulary(Vocabulary):
    def __init__(self, args: Namespace):
        super().__init__()

        with open(args.dict_path, "rb") as f:
            subtoken_count = pickle.load(f)
            node_count = pickle.load(f)
            target_count = pickle.load(f)

        self.subtoken_to_index, self.index_to_subtoken = index_mapping_from_dict(
            subtoken_count, additional_keys=[self.PAD_TOKEN]
        )
        self.target_to_index, self.index_to_target = index_mapping_from_dict(
            target_count,
            additional_keys=[self.PAD_TOKEN, self.EOS_TOKEN, self.SOS_TOKEN],
        )
        self.node_to_index, self.index_to_target = index_mapping_from_dict(
            node_count,
            additional_keys=[self.PAD_TOKEN],
        )

    def target_vocab_size(self):
        return len(self.target_to_index)

    def node_vocab_size(self):
        return len(self.node_to_index)

    def terminal_vocab_size(self):
        return len(self.subtoken_to_index)

    def add_special_target_token(self, token: str):
        self.target_encoder.add_special_tokens(
            self.target_encoder.special_tokens + [token]
        )

    def add_special_terminal_token(self, token: str):
        self.subtoken_encoder.add_special_tokens(
            self.subtoken_encoder.special_tokens + [token]
        )

    def encode_node(self, token_or_tokens):
        if isinstance(token_or_tokens, str):
            return self.node_to_index[token_or_tokens]
        else:
            return list(map(self.encode_node, token_or_tokens))

    def decode_node(self, index_or_indices):
        if isinstance(index_or_indices, int):
            return self.index_to_node[index_or_indices]
        else:
            return list(map(self.decode_node, index_or_indices))

    def encode_target(self, token_or_tokens):
        if isinstance(token_or_tokens, str):
            return self.target_to_index.token_to_id(token_or_tokens)
        else:
            return list(map(self.encode_target, token_or_tokens))

    def decode_target(self, index_or_indices):
        if isinstance(index_or_indices, int):
            return self.index_to_target[index_or_indices]
        else:
            return list(map(self.decode_target, index_or_indices))

    def encode_terminal(self, token_or_tokens):
        if isinstance(token_or_tokens, str):
            return self.subtoken_to_index.token_to_id(token_or_tokens)
        else:
            return list(map(self.encode_terminal, token_or_tokens))

    def decode_terminal(self, index_or_indices):
        if isinstance(index_or_indices, int):
            return self.index_to_subtoken[index_or_indices]
        else:
            return list(map(self.decode_terminal, index_or_indices))
