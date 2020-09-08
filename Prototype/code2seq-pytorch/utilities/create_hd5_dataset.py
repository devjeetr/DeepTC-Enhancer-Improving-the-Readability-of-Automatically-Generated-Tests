import argparse
import os

def cmd_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--i", required=True)

    return parser.parse_args()


def read_in_chunks(file_object, chunk_size=1024):
    """Lazy function (generator) to read a file piece by piece.
    Default chunk size: 1k."""
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data



def process_file(filepath):
    subtoken_count = {}
    node_count = {}
    target_count = {}

    with open(filepath, "r") as f:
        for line in f:
            line_parts = line.strip().split(" ")
            label = line_parts[0]

            for token in label.split("|"):
                target_count[token] = target_count.get(token, 0) + 1

            for context in line_parts[1:]:
                context_parts = context.split(",")
                if len(context_parts) != 3:
                    continue
                start, nodes, end = context_parts

                for token in start.split("|"):
                    subtoken_count[token] = subtoken_count.get(token, 0) + 1

                for token in end.split("|"):
                    subtoken_count[token] = subtoken_count.get(token, 0) + 1

                for node in nodes.split("|"):
                    node_count[node] = node_count.get(node, 0) + 1
                # for df in pd.read_csv(filepath, header=None, )

    return subtoken_count, node_count, target_count

def write_dict(mapping, filename):
    with open(filename, "w") as f:
        for k, v in mapping.items():
            f.write(f"{k}|{v}\n")


def main():
    args = cmd_args()

    (subtoken_count, node_count, target_count) = process_file(args.i)

    base_dir = os.path.dirname(args.i)

    write_dict(subtoken_count, os.path.join(base_dir, "subtoken_count.txt"))
    write_dict(target_count, os.path.join(base_dir, "target_count.txt"))
    write_dict(node_count, os.path.join(base_dir, "node_count.txt"))


if __name__ == "__main__":
    main()
