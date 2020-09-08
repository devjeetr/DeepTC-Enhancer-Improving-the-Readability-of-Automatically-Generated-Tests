import torch
import random

from utilities.common import (
    extract_training_example_from_line,
    encode_tokens,
    decompose_context,
)


def pad_arr(arr, n, pad):
    if len(arr) > n:
        raise Exception("Length of input array greater than pad size")
    pad_len = n - len(arr)
    return arr + [pad for _ in range(pad_len)]


def is_context_empty(context: list) -> bool:
    return (len(context[0]) == 0) and (len(context[1]) == 0) and (len(context[2]) == 0)


def sample_contexts(contexts: list, n: int) -> list:
    """ samples n contexts from contexts. If length of contexts < n,
        no sampling is performed.
    """
    return random.sample(contexts, n) if len(contexts) > n else contexts


def pad_contexts(contexts, n):
    if len(contexts) < n:
        pad_len = n - len(contexts)
        contexts = contexts + [[[], [], []] for _ in range(pad_len)]

    return contexts


def context_encodings_to_tensors(
    contexts,
    max_contexts,
    max_token_len,
    max_ast_len,
    subtoken_pad_idx,
    ast_pad_idx,
    shuffle=False,
):
    """Preps context encodings to create tensors ready for training

        Args:
            contexts : an array of shape (n_contexts * 3 * )
            max_contexts: the max number of contexts
            max_token_len: max number of sub-tokens for start/end
            max_ast_len: max length of each path in the context
            subtoken_pad_idx: integer to use to pad subtoken tensor
            ast_pad_idx: integer to use to pad ast path tensor
            shuffle: whether or not to shuffle the ordering of the contexts
    """
    contexts = sample_contexts(contexts, max_contexts)
    contexts = pad_contexts(contexts, max_contexts)
    if shuffle:
        random.shuffle(contexts)

    mask = [0 if is_context_empty(context) else 1 for context in contexts]

    ast_lengths = [len(context[1]) for context in contexts]
    start_lengths = [len(context[0]) for context in contexts]
    end_lengths = [len(context[2]) for context in contexts]

    start = [
        pad_arr(context[0], max_token_len, subtoken_pad_idx) for context in contexts
    ]
    end = [pad_arr(context[2], max_token_len, subtoken_pad_idx) for context in contexts]
    paths = [pad_arr(context[1], max_ast_len, ast_pad_idx) for context in contexts]

    return (
        torch.tensor(start),
        torch.tensor(end),
        torch.tensor(paths),
        torch.tensor(mask),
        torch.tensor(start_lengths),
        torch.tensor(end_lengths),
        torch.tensor(ast_lengths),
    )


def clip_arr_length(arr, n):
    if n != -1 and len(arr) > n:
        arr = arr[:n]
    return arr


def preprocess_example(
    line,
    subtoken_to_index,
    node_to_index,
    target_to_index,
    subtoken_len=-1,
    ast_path_len=-1,
    target_len=-1,
    eos_idx=None,
    sos_idx=None,
    relevant_only=True,
):
    """ Given a raw line in code2seq format, this function:
        1. Extracts label and contexts
        2. Decomposes each context
        3. Encodes the subtokens and ast paths
        4. Encodes the label subtokens

        Return:
            (encoded_labels, encoded_contexts)
    """
    label, contexts = extract_training_example_from_line(line)
    encoded_label = encode_tokens(label.split("|"), target_to_index)

    encoded_label = clip_arr_length(encoded_label, target_len)

    if eos_idx is not None:
        encoded_label.append(eos_idx)
    if sos_idx is not None:
        encoded_label.insert(0, sos_idx)

    encoded_contexts = []
    for context in contexts:
        start_node, ast_path, end_node = decompose_context(context)

        if (
            relevant_only
            and "TARGET_VARIABLE" not in start_node
            and "TARGET_VARIABLE" not in end_node
        ):
            continue

        # clip lengths
        start_node = clip_arr_length(start_node, subtoken_len)
        end_node = clip_arr_length(end_node, subtoken_len)
        ast_path = clip_arr_length(ast_path, ast_path_len)

        start_node, ast_path, end_node = map(
            lambda x: encode_tokens(x[0], x[1]),
            zip(
                [start_node, ast_path, end_node],
                [subtoken_to_index, node_to_index, subtoken_to_index],
            ),
        )

        encoded_contexts.append([start_node, ast_path, end_node])

    assert len(contexts) > 0

    return encoded_label, encoded_contexts
