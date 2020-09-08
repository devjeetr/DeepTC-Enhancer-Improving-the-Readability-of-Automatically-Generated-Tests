from preprocessing.chunked_pmap import chunked_pmap
from spiral import ronin
import argparse
from preprocessing.context_utils import (
    path_iterator,
    is_variable,
    is_name_expr,
    get_end_node,
    get_start_node,
    get_variables_from_contexts,
)
import cursor
import os

cursor.show()


def add_subtokens_to_dict(subtokens, _dict, split_subtokens=False):
    subtokens = subtokens.strip().split("|")

    for subtoken in subtokens:
        _dict[subtoken] = _dict.get(subtoken.strip(), 0) + 1


def process_chunk(lines):
    target_count = {}
    subtoken_count = {}
    node_count = {}

    for label, contexts in path_iterator(lines):
        add_subtokens_to_dict(label, target_count, split_subtokens=True)
        for start, path, end in contexts:
            add_subtokens_to_dict(start, subtoken_count, split_subtokens=True)
            add_subtokens_to_dict(end, subtoken_count, split_subtokens=True)
            add_subtokens_to_dict(path, node_count)

    return target_count, subtoken_count, node_count


def should_mask(token, node, variables):
    return token in variables and (is_variable(node) or is_name_expr(node))


def process_variables(lines):
    target_count, subtoken_count, node_count = {}, {}, {}

    for label, contexts in path_iterator(lines):

        variables = get_variables_from_contexts(contexts)
        for variable in variables:
            add_subtokens_to_dict(variable, target_count)

        for start, path, end in contexts:
            start_node, end_node = [f(path) for f in [get_start_node, get_end_node]]
            if should_mask(start, start_node, variables):
                add_subtokens_to_dict(start, subtoken_count)

            if not should_mask(end, end_node, variables):
                add_subtokens_to_dict(end, subtoken_count)

            add_subtokens_to_dict(path, node_count)

    return target_count, subtoken_count, node_count


def add_counts(original, to_add):
    for k, v in to_add.items():
        original[k] = original.get(k, 0) + v


def write_dict(_dict, outfile):
    with open(outfile, "w") as f:
        items = [f"{k}|{v}" for k, v in _dict.items()]
        f.write("\n".join(items))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--i", help="The c2s file for which to construct the vocabulary"
    )
    parser.add_argument("--o", help="The output folder")
    parser.add_argument("--variables", action="store_true")
    parser.add_argument("--chunk-size", default=2 ** 29)
    parser.add_argument("--n-workers", default=12, type=int)
    parser.add_argument(
        "--prefix", help="prefix for token frequency files", default=None
    )

    args = parser.parse_args()

    prefix = f"{args.prefix}." if args.prefix else ""

    node_counts_file = f"{prefix}node_counts.txt"
    subtoken_counts_file = f"{prefix}subtoken_counts.txt"
    target_counts_file = f"{prefix}target_counts.txt"

    base_dir = args.o if args.o else os.path.dirname(args.i)
    f = process_variables if args.variables else process_chunk
    results = chunked_pmap(
        args.i, f, reducer=True, file_chunk_size=2 ** 27, n_futures=args.n_workers
    )

    target_count = {}
    subtoken_count = {}
    node_count = {}


    for entry in results:
        add_counts(target_count, entry[0])
        add_counts(subtoken_count, entry[1])
        add_counts(node_count, entry[2])
    print("-" * 60)
    print(f"\tDictionary sizes:")
    print(f"\tsubtokens: {len(subtoken_count)}")
    print(f"\tTarget: {len(target_count)}")

    write_dict(target_count, os.path.join(base_dir, target_counts_file))
    write_dict(subtoken_count, os.path.join(base_dir, subtoken_counts_file))
    write_dict(node_count, os.path.join(base_dir, node_counts_file))


if __name__ == "__main__":
    main()
