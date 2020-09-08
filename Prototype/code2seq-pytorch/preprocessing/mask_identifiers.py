import argparse
import os
from preprocessing.chunked_pmap import chunked_pmap
from preprocessing.context_utils import path_iterator, mask_variables_in_contexts


def cmd_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--i", help="The input file containing paths (c2s format)")
    parser.add_argument("--o", help="The output file")

    return parser.parse_args()


def process_lines(lines):
    results = []
    for label, contexts in path_iterator(lines):
        masked_contexts = mask_variables_in_contexts(contexts)
        masked_contexts = [
            ",".join(masked_context) for masked_context in masked_contexts
        ]
        contexts_str = " ".join(masked_contexts)

        results.append(f"{label} {contexts_str}\n")

    return results


def write_chunks(f):
    def write_chunks_to_current_file(results):
        for result in results:
            for line in result:
                f.write(line)

    return write_chunks_to_current_file


def process_file(infile, outfile):
    with open(outfile, "w") as out_f:
        on_receive = write_chunks(out_f)
        chunked_pmap(infile, process_lines, on_receive_chunk=on_receive)


def main():
    args = cmd_args()
    out_dir = os.path.dirname(args.o)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    process_file(args.i, args.o)


if __name__ == "__main__":
    main()
