import argparse
import os
from tokenizers import SentencePieceBPETokenizer
import tempfile
from math import ceil


def cmd_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--i", required=True)
    parser.add_argument("--o")
    parser.add_argument("--n", required=True)
    parser.add_argument("--vocab-size", default=30000, type=int)
    return parser.parse_args()


def create_bpe_training_file(count_file, outfile):
    with open(outfile, "w") as out:
        with open(count_file, "r") as infile:
            counts = {}
            print(f"Parsing counts file")
            for line in infile:
                word, count = line.split("|")
                count = int(count)
                counts[word] = count
            for i, (k, v) in enumerate(counts.items()):
                rectified_count = v

                for _ in range(rectified_count):
                    out.write(f"{k}\n")
                print(f"Written {i+1} / {len(counts)} entries", end="\r")

def main():
    args = cmd_args()
    outdir = args.o if args.o else os.path.dirname(args.i)

    print(f"Training SentencePiece to create a vocabulary of size {args.vocab_size}")
    with tempfile.TemporaryDirectory() as tmp_dir:
        train_file = os.path.join(tmp_dir, "train.txt")
        create_bpe_training_file(args.i, train_file)

        tokenizer = SentencePieceBPETokenizer()
        tokenizer.train(files=[train_file], vocab_size=args.vocab_size)

    tokenizer.save(outdir, args.n)


if __name__ == "__main__":
    main()
