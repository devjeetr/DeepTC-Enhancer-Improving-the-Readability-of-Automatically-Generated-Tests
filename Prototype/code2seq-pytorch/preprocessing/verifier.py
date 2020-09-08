from preprocessing.chunked_pmap import chunked_pmap
from preprocessing.context_utils import (
    path_iterator,
    is_variable,
    is_name_expr,
    get_start_node,
    get_end_node,
)
from argparse import ArgumentParser
import ray


def cmd_args():
    parser = ArgumentParser()
    parser.add_argument("--i")
    parser.add_argument("--variable", action="store_true")

    return parser.parse_args()


@ray.remote
def verify_variable_dataset(lines):

    for variable, contexts in path_iterator(lines):
        # make sure that if variable appears in any context,
        # it is not a VDID or a Nm

        for context in contexts:
            start, path, end = map(lambda x: x.strip(), context)
            if start == variable:
                start_node = get_start_node(path)
                assert not (
                    is_name_expr(start_node) or is_variable(start_node)
                ), f"Verification Failed: Variable appears in start node and is not a variable or a name expr:\n{context}"
                # print(f"variable name appeared with {start_node}")
            if end == variable:
                end_node = get_end_node(path)
                assert not (
                    is_name_expr(end_node) or is_variable(end_node)
                ), f"Verification Failed: Variable appears in start node and is not a variable or a name expr:\n{context}"

                # print(f"variable name appeared with {end_node}")


def main():

    args = cmd_args()
    ray.init()

    chunked_pmap(
        args.i, verify_variable_dataset, on_receive_chunk=lambda x: None, ray=ray
    )

    ray.shutdown()

if __name__ == "__main__":
    main()
