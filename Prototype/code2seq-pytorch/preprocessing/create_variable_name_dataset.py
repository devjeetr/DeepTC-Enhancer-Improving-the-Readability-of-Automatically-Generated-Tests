import argparse
from argparse import Namespace
from typing import Iterable, List, Any
from preprocessing.chunked_pmap import chunked_pmap
from preprocessing.context_utils import (
    path_iterator,
    get_variables_from_contexts,
    get_start_node,
    get_end_node,
    mask_variables_in_contexts,
    is_variable,
    is_name_expr,
    mask_context,
)


def cmd_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--i",
        help="path to the method name c2s file from which to generate variable name dataset",
    )
    parser.add_argument("--o", help="path to store output in")
    parser.add_argument(
        "--n-workers", help="number of ray remotes", default=12, type=int
    )

    return parser.parse_args()


def compute_mask(token: str, node: str, target: str, variables: List[str]) -> bool:
    if token in variables and (is_variable(node) or is_name_expr(node)):
        if token == target:
            return "TARGET_VARIABLE"
        else:
            return "VARIABLE"

    return token


def create_path_for_variable(
    variable: str, variables: List[str], contexts: List[Any]
) -> Iterable[str]:
    masked_contexts = mask_variables_in_contexts(
        contexts, target_variable=variable, variables=variables, keep_only_target=True
    )
    masked_contexts = [",".join(masked_context) for masked_context in masked_contexts]
    masked_contexts = " ".join(masked_contexts)

    return f"{variable} {masked_contexts}"


def process_chunk(lines: List[str]):
    masked_entries = []
    for label, contexts in path_iterator(lines):
        variables = get_variables_from_contexts(contexts)
        for variable in variables:
            new_entries = create_path_for_variable(variable, variables, contexts)
            new_entries = new_entries.strip()

            assert "\n" not in new_entries, "Newline found in new_entries"

            if len(new_entries) > 0:
                masked_entries.append(new_entries)

    return masked_entries


def process_chunk_serial(lines: List[str]):
    masked_entries = []
    for label, contexts in path_iterator(lines):
        variables = get_variables_from_contexts(contexts)
        for variable in variables:
            new_entries = create_path_for_variable(variable, variables, contexts)
            new_entries = new_entries.strip()

            assert "\n" not in new_entries, "Newline found in new_entries"

            if len(new_entries) > 0:
                masked_entries.append(new_entries)

    return masked_entries


def process_file(
    input_file, output_file, n_workers,
):
    with open(output_file, "w") as f:

        def write_chunk(chunk):
            for lines in chunk:
                for line in lines:
                    f.write(line + "\n")

        chunked_pmap(
            input_file,
            process_chunk,
            on_receive_chunk=write_chunk,
            file_chunk_size=2 ** 28,
            n_futures=n_workers,
        )


def process_file_sequential(input_file, output_file):
    with open(input_file, "r") as inp:
        with open(output_file, "w") as out:
            results = process_chunk_serial(inp.readlines())
            for line in results:
                out.write(line.strip() + "\n")


def generate_variable_name_data(args):
    if args.n_workers == 1:
        process_file_sequential(args.i, args.o)
    else:
        process_file(args.i, args.o, n_workers=args.n_workers)


def main():
    args = cmd_args()
    generate_variable_name_data(args)


if __name__ == "__main__":
    main()
