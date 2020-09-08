import argparse
from itertools import chain
import os
from tokenizers import SentencePieceBPETokenizer
import tempfile
from math import ceil
from preprocessing.context_utils import path_iterator
from preprocessing.chunked_pmap import chunked_pmap
from utilities.data.Vocabulary import Vocabulary
from enum import Enum


class Preset(Enum):
    variable = "variable"
    method = "method"

    def __str__(self):
        return self.value


def get_special_tokens(preset: Preset):
    if preset == Preset.variable:
        subtoken_special_tokens = [
            "METHOD_NAME",
            "TARGET_VARIABLE",
            "VARIABLE",
            Vocabulary.PAD_TOKEN,
        ]
    else:
        subtoken_special_tokens = ["METHOD_NAME", "VARIABLE", Vocabulary.PAD_TOKEN]

    target_special_tokens = [
        Vocabulary.PAD_TOKEN,
        Vocabulary.SOS_TOKEN,
        Vocabulary.EOS_TOKEN,
        Vocabulary.UNK_TOKEN,
    ]

    return target_special_tokens, subtoken_special_tokens


def cmd_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--i", required=True)
    parser.add_argument("--o")
    parser.add_argument("--subtoken-vocab", default=30000, type=int)
    parser.add_argument("--target-vocab", default=16000, type=int)
    parser.add_argument("--preset", type=Preset, choices=list(Preset), required=True)

    return parser.parse_args()


def extract_targets_and_subtokens(lines):
    subtokens = []
    targets = []

    for target, contexts in path_iterator(lines):
        targets.append(" ".join(target.split("|")))

        for start, _, end in contexts:
            subtokens.append(" ".join(start.split("|")))
            subtokens.append(" ".join(end.split("|")))

    return {"subtokens": subtokens, "targets": targets}


def flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i

def create_bpe_training_file(dataset, targets_file_path, subtokens_file_path):
    with open(targets_file_path, "w") as f_t, open(subtokens_file_path, "w") as f_s:

        def write_to_file(chunks):
            for entry in flatten(chunks):
                f_t.write("\n".join(entry["targets"]))
                f_s.write("\n".join(entry["subtokens"]))

        chunked_pmap(
            dataset,
            mapper=extract_targets_and_subtokens,
            on_receive_chunk=write_to_file,
        )


def main():
    args = cmd_args()
    outdir = args.o if args.o else os.path.dirname(args.i)

    target_special_tokens, subtoken_special_tokens = get_special_tokens(args.preset)
    with tempfile.TemporaryDirectory() as tmp_dir:
        targets_file = os.path.join(tmp_dir, "labels.txt")
        subtokens_file = os.path.join(tmp_dir, "subtokens.txt")

        print(f"Creating training files for BPE")
        create_bpe_training_file(args.i, targets_file, subtokens_file)
        if args.preset == Preset.variable:
            print("Variable preset")

        subtoken_tokenizer = SentencePieceBPETokenizer()
        target_tokenizer = SentencePieceBPETokenizer()
        print(f"Training subtoken tokenizer")
        subtoken_tokenizer.add_special_tokens(subtoken_special_tokens)
        print(f"Training target tokenizer")
        target_tokenizer.add_special_tokens(target_special_tokens)

        target_tokenizer.train(files=[targets_file], vocab_size=args.target_vocab)
        subtoken_tokenizer.train(files=[subtokens_file], vocab_size=args.subtoken_vocab)

    target_tokenizer.save(outdir, "target.bpe")
    subtoken_tokenizer.save(outdir, "subtoken.bpe")


if __name__ == "__main__":
    main()
