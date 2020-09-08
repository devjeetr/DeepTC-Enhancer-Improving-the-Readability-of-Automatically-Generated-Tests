from typing import NamedTuple, Iterator, Collection


def path_iterator(lines, split_path=False, split_terminals=False):
    """ A generator used to process raw contexts from the c2s file.
    """
    for line in lines:
        line_parts = line.split(" ")
        if len(line_parts) < 2:
            continue

        label = line_parts[0].strip()

        contexts = []
        for context in line_parts[1:]:
            context_parts = context.split(",")
            if len(context_parts) != 3:
                continue

            start, path, end = context_parts
            start, path, end = [s.strip() for s in [start, path, end]]
            if split_path:
                path = path.split("|")

            if split_terminals:
                start = start.split("|")
                end = end.split("|")

            del context_parts
            contexts.append((start.strip(), path.strip(), end.strip()))

        yield label, contexts


def is_start_variable(path):
    return path.startswith("VDID")


def is_end_variable(path: str):
    return get_end_node(path).startswith("VDID")


def is_name_expr(node: str):
    return node.startswith("Nm")


def is_variable(node: str) -> bool:
    return node.startswith("VDID")


def is_identifier_a_variable(
    identifier: str, node_type: str, variables: Collection[str]
):
    return identifier in variables and (
        is_name_expr(node_type) or is_variable(node_type)
    )


GENERAL_VARIABLE_MASK = "VARIABLE"
TARGET_VARIABLE_MASK = "TARGET_VARIABLE"


def get_variable_mask(
    variable,
    target_variable,
    general_variable_mask=GENERAL_VARIABLE_MASK,
    target_variable_mask=TARGET_VARIABLE_MASK,
):
    if variable == target_variable:
        return target_variable_mask

    return general_variable_mask


def mask_variables_in_context(
    context,
    variables,
    mask=GENERAL_VARIABLE_MASK,
    target_variable=None,
    target_variable_mask=TARGET_VARIABLE_MASK,
):
    start, path, end = context

    if target_variable:
        assert (
            target_variable in variables
        ), "Provided target variable is not in variables"

    start_node = get_start_node(path)
    end_node = get_end_node(path)

    start_mask, end_mask = [
        get_variable_mask(
            identifier,
            target_variable,
            general_variable_mask=mask,
            target_variable_mask=target_variable_mask,
        )
        if is_identifier_a_variable(identifier, node_type, variables)
        else None
        for identifier, node_type in ((start, start_node), (end, end_node))
    ]

    return mask_context(context, start_mask=start_mask, end_mask=end_mask)


def mask_variables_in_contexts(
    contexts, target_variable=None, variables=None, keep_only_target=False
):
    if not variables:
        variables = get_variables_from_contexts(contexts)
    masked_contexts = []

    for context in contexts:
        masked_start, masked_path, masked_end = mask_variables_in_context(
            context, variables, target_variable=target_variable
        )
        if (
            keep_only_target
            and TARGET_VARIABLE_MASK != masked_start
            and TARGET_VARIABLE_MASK != masked_end
        ):
            # print(f"Skipping {masked_start} -> {masked_end}")
            continue
        masked_contexts.append((masked_start, masked_path, masked_end))

    return masked_contexts


def get_variables_from_contexts(contexts):
    variables = set()

    for start, path, end in contexts:
        if is_start_variable(path):
            variables.add(start)
        if is_end_variable(path):
            variables.add(end)

    return variables


def get_start_node(path):
    return path.split("|", 1)[0]


def get_end_node(path):
    return path.rsplit("|", 1)[-1]


def mask_context(context, start_mask=None, end_mask=None):
    """ Masks the given context using the given start and end masks, if provided.
        If no masks are provided, returns the original context unchanged.

    """
    start, path, end = context
    return (
        start_mask if start_mask is not None else start,
        path,
        end_mask if end_mask is not None else end,
    )
