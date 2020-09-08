import os
import argparse
from tokenizers import SentencePieceBPETokenizer
from preprocessing.context_utils import path_iterator
from preprocessing.chunked_pmap import chunk_list
from multiprocessing import Pool


def compute_lengths(lines, subtoken_tokenizer, target_tokenizer):
    target_lengths = []
    subtoken_lengths = []
    for label, contexts in path_iterator(lines):
        target_lengths.append(len(target_tokenizer.encode(label.split("|"))))

        for start, path, end in contexts:
            subtoken_lengths.append(len(subtoken_tokenizer.encode(start.split("|"))))
            subtoken_lengths.append(len(subtoken_tokenizer.encode(end.split("|"))))

    return target_lengths, subtoken_lengths

import sys
def print_stats_for_file(data_file, subtoken_tokenizer, target_tokenizer):
    pool = Pool(n_processes=12)

    with open(data_file, "r") as f:
        while True:
            for lines in f.readlines():
                results = pool.starmap(compute_lengths, [(lines, subtoken_tokenizer, target_tokenizer,) for chunks in chunk_list(lines, )])


def main():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--subtoken-prefix", default="subtoken.bpe")
    parser.add_argument("--target-prefix", default="target.bpe")

    args = parser.parse_args()

    main()
