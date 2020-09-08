import argparse
import pickle
import numpy as np
import os

def cmd_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", help="directory with the .txt files")
    parser.add_argument("--node-count", default="node_count.txt")
    parser.add_argument("--target-count", default="target_count.txt")
    parser.add_argument("--subtoken-count", default="subtoken_count.txt")
    parser.add_argument("--threshold", default=0.25)
    parser.add_argument("--o", default=None)

    return parser.parse_args()


def load_counts(filename):
    freqs = {}
    i = 0
    with open(filename, "r") as f:
        for line in f:
            token, count = line.split("|")
            freqs[token] = int(count)
            i += 1
    return freqs

def trim_by_quantile(kv, threshold=0.25):
    q = np.quantile(np.array(list(kv.values())), 0.25)
    return {k:v for k,v in kv.items() if v > q}


def main():
    args = cmd_args()

    target_count = load_counts(os.path.join(args.dir, args.target_count))
    node_count = load_counts(os.path.join(args.dir, args.node_count))
    subtoken_count = load_counts(os.path.join(args.dir, args.subtoken_count))

    target_count = trim_by_quantile(target_count)
    node_count = trim_by_quantile(node_count)
    subtoken_count = trim_by_quantile(subtoken_count)


    outfile_path = args.o if args.o is not None else os.path.join(args.dir, "dict.c2s")

    with open(outfile_path, 'wb') as file:
        pickle.dump(subtoken_count, file)
        pickle.dump(node_count, file)
        pickle.dump(target_count, file)
        pickle.dump(200, file)
        pickle.dump(57000, file)

    print()
    print(f"Successfully saved to {outfile_path}")
    print()

if __name__ == "__main__":
    main()