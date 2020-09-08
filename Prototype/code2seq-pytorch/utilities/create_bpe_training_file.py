import argparse
import os
from tokenizers import SentencePieceBPETokenizer
import numpy as np


def cmd_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--i", required=True)
    parser.add_argument("--o")
    parser.add_argument("-n", required=True)
    return parser.parse_args()


from collections import Counter
from math import ceil


def main():
    args = cmd_args()
    outdir = args.o if args.o else os.path.dirname(args.i)

    outfile = os.path.join(outdir, args.n)
    print("starting!")
    with open(outfile, "w") as out:
        with open(args.i, "r") as infile:
            min_count = float("inf")
            counts = {}
            for line in infile:
                word, count = line.split("|")
                count = int(count)
                counts[word] = count

            for k, v in counts.items():
                rectified_count = ceil(v / min_count)

                for _ in range(rectified_count):
                    out.write(f"{k}\n")



if __name__ == "__main__":
    main()
