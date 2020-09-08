from abc import ABC, abstractmethod
from typing import Union, List


class Vocabulary(ABC):
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    SOS_TOKEN = "<S>"
    EOS_TOKEN = "</S>"

    def __init__(self, subtoken_len=-1, ast_len=-1, target_len=-1):
        pass
    

    @abstractmethod
    def target_vocab_size(self):
        pass

    @abstractmethod
    def node_vocab_size(self):
        pass

    @abstractmethod
    def terminal_vocab_size(self):
        pass

    @abstractmethod
    def encode_terminal(self, token_or_tokens: Union[str, List[str]]):
        """ Encodes the given token or list of tokens.

            :param token_or_tokens either a token or a list of tokens
            :return encoded tokens
        """
        pass

    @abstractmethod
    def decode_terminal(
        self, index_or_indices: Union[int, List[int]]
    ) -> Union[str, List[str]]:
        pass

    @abstractmethod
    def encode_target(
        self, token_or_tokens: Union[str, List[str]]
    ) -> Union[int, List[int]]:
        """
        """
        pass

    @abstractmethod
    def decode_target(
        self, index_or_indices: Union[int, List[int]]
    ) -> Union[str, List[str]]:
        """ Decodes the given target index or indices
            :param index_or_indices a string or list of strings
            :return a string or a list of strings
            :throws ValueError if any of the provided indices are not in the target vocabulary
        """
        pass

    @abstractmethod
    def encode_node(
        self, token_or_tokens: Union[str, List[str]]
    ) -> Union[int, List[int]]:
        """ Encodes the given node or list of nodes
            :param token_or_tokens a string or list of strings
            :return an integer or list of integers
        """
        pass

    @abstractmethod
    def decode_node(
        self, index_or_indices: Union[int, List[int]]
    ) -> Union[str, List[str]]:
        """ Decodes the given encoded node or nodes
            :param index_or_indices an integer or list of integers
            :return a string or a list of strings
        """
        pass
