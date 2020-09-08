from torch import is_tensor
from torch.utils.data import Dataset
import torch
import random
from utilities.config import Config
from utilities.FileReader import FileReader
from utilities.preprocessing import context_encodings_to_tensors, preprocess_example
from utilities.preprocessing import pad_arr
from utilities.data import Vocabulary
from argparse import Namespace
from typing import Tuple
import numpy as np


def get_contexts_containing_target_variable(contexts):

    return [
        context
        for context in contexts
        if "TARGET_VARIABLE" in context.split(",", 1)[0].strip()
        or "TARGET_VARIABLE" in context.rsplit(",", 1)[-1].strip()
    ]


def line_to_example(
    line: str,
    vocabulary: Vocabulary,
    subtoken_len: int,
    target_len: int,
    ast_len: int,
    max_contexts: int,
    shuffle=False,
    contexts_should_contain_target_variable=False,
) -> Tuple[np.array, Tuple[np.array, np.array, np.array]]:
    """ Takes a raw line from a c2s file and returns encoded numpy vectors
    """
    line_parts = line.split(" ")

    label = line_parts[0]
    raw_label_encoding = vocabulary.encode_target(
        [vocabulary.SOS_TOKEN] + label.split("|")[:target_len] + [vocabulary.EOS_TOKEN],
    )[: target_len + 2]

    encoded_label = np.full(
        (target_len + 2), vocabulary.encode_target(vocabulary.PAD_TOKEN)
    )
    encoded_label[: len(raw_label_encoding)] = np.array(raw_label_encoding)

    starts = np.full(
        (max_contexts, subtoken_len), vocabulary.encode_terminal(vocabulary.PAD_TOKEN)
    )
    ends = np.full(
        (max_contexts, subtoken_len), vocabulary.encode_terminal(vocabulary.PAD_TOKEN)
    )
    paths = np.full(
        (max_contexts, ast_len), vocabulary.encode_node(vocabulary.PAD_TOKEN)
    )

    # print(paths.shape)
    contexts = line_parts[1:]

    contexts = (
        get_contexts_containing_target_variable(contexts)
        if contexts_should_contain_target_variable
        else contexts
    )

    if shuffle:
        random.shuffle(contexts)
    else:
        raise Exception("u must shuffle the contexts")

    for i, context in enumerate(contexts[:max_contexts]):
        start, path, end = context.split(",")

        # print(f"\n{start} -> {end}\n")
        # print(start)
        start = vocabulary.encode_terminal(start.split("|"))[:subtoken_len]

        end = vocabulary.encode_terminal(end.split("|"))[:subtoken_len]
        path = vocabulary.encode_node(path.split("|"))[:ast_len]

        starts[i, : len(start)] = np.array(start)
        ends[i, : len(end)] = np.array(end)
        paths[i, : len(path)] = np.array(path)

    return encoded_label, (starts, ends, paths)


class C2SDataSet(Dataset):
    def __init__(
        self,
        data_file: str,
        max_contexts: int,
        vocabulary: Vocabulary,
        subtoken_len: int,
        ast_path_len: int,
        target_len: int,
        shuffle=True,
        variable_only_filter=True,
        line_cache=None,
    ):
        self.vocabulary = vocabulary
        self.subtoken_len = subtoken_len
        self.ast_path_len = ast_path_len
        self.target_len = target_len
        self.shuffle = shuffle
        self.max_contexts = max_contexts
        self.variable_only_filter = variable_only_filter
        self.reader = FileReader(data_file, line_cache=line_cache)

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()

        label, (start, end, path) = line_to_example(
            self.reader[idx],
            self.vocabulary,
            self.subtoken_len,
            self.target_len,
            self.ast_path_len,
            self.max_contexts,
            shuffle=self.shuffle,
            contexts_should_contain_target_variable=self.variable_only_filter,
        )

        start_lengths = (
            start != self.vocabulary.encode_terminal(self.vocabulary.PAD_TOKEN)
        ).sum(axis=1)
        end_lengths = (
            end != self.vocabulary.encode_terminal(self.vocabulary.PAD_TOKEN)
        ).sum(axis=1)
        ast_lengths = (
            path != self.vocabulary.encode_node(self.vocabulary.PAD_TOKEN)
        ).sum(axis=1)

        context_masks = 1 * (ast_lengths != 0)

        label, start, end, path = map(torch.tensor, (label, start, end, path))
        start_lengths, end_lengths, ast_lengths = map(
            torch.tensor, (start_lengths, end_lengths, ast_lengths)
        )

        assert start_lengths.sum() != 0, f"start lengths should not be null"

        return (
            label,
            (start, end, path, context_masks, start_lengths, end_lengths, ast_lengths),
        )

    def __len__(self):
        return len(self.reader)
