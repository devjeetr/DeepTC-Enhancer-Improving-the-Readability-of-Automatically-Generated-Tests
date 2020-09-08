import pickle
from utilities.config import Vocabularies

def extract_training_example_from_line(line):
    """Given a line from a training file(in code2seq format),
       extracts and returns the label and context

       Return:
            (label, contexts)
    """
    fields = line.split(" ")
    label = fields[0]
    contexts = fields[1:]

    return label, contexts


def encode_tokens(strings, mapping, unknown_key="<UNK>"):
    return [mapping.get(string, mapping[unknown_key]) for string in strings]


def decompose_context(context):
    """ Given a raw context string in code2seq format,
        decomposes it into a 3 tuple of start, ast_nodes, end

        Return:
            (start, ast_path, end)
    """
    context_parts = context.split(",")
    if len(context_parts) < 3:
        return "", "", ""

    start_node = context_parts[0].split("|")
    end_node = context_parts[2].split("|")
    ast_path = context_parts[1].split("|")

    return start_node, ast_path, end_node


def getUniqueNodesInLine(line):
    contexts = line.split(" ")[1:]
    unique_nodes = set()
    for context in contexts:
        context_parts = context.split(",")
        if len(context_parts) == 3:
            nodes = context_parts[1].split("|")
            for node in nodes:
                unique_nodes.add(node)

    return unique_nodes


def createNodeIndexMapping(filename, max_lines=None):
    node_to_index, index_to_node = {}, {}

    current_index = 0
    with open(filename, "r+") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i > max_lines:
                return node_to_index, index_to_node

            if i % 1000 == 0:
                print(f"Processing line {i + 1}, vocab size: {len(node_to_index)}")
            nodes = getUniqueNodesInLine(line)
            for node in nodes:
                if node not in node_to_index:
                    node_to_index[node] = current_index
                    index_to_node[current_index] = node
                    current_index += 1

    return node_to_index, index_to_node


def createIndexMappingFromDict(dict, max_size=-1, additional_keys={}):
    mapping, inverse = {}, {}

    counter = 0

    for key in additional_keys:
        mapping[key] = counter
        inverse[counter] = key
        counter += 1

    for key in dict:
        mapping[key] = counter
        inverse[counter] = key
        counter += 1

        if max_size != -1 and counter == max_size:
            break

    return mapping, inverse


def load_dict(dict_root):
    with open(dict_root, "rb") as file:
        subtoken_to_count = pickle.load(file)
        node_to_count = pickle.load(file)
        target_to_count = pickle.load(file)
        max_contexts = pickle.load(file)
        num_training_examples = pickle.load(file)

    return (
        subtoken_to_count,
        node_to_count,
        target_to_count,
        max_contexts,
        num_training_examples,
    )


def create_vocabularies_from_frequency_dict(
    dict_file_path, subtoken_special_keys, node_special_keys, target_special_keys
):
    """

        Returns:
            (
                (subtoken_to_index, index_to_subtoken),
                (node_to_index, index_to_node),
                (target_to_index, index_to_target),
            )
    """
    (
        subtoken_to_count,
        node_to_count,
        target_to_count,
        max_contexts,
        num_training_examples,
    ) = load_dict(dict_file_path)

    subtoken_to_index, index_to_subtoken = createIndexMappingFromDict(
        subtoken_to_count, additional_keys=subtoken_special_keys
    )
    node_to_index, index_to_node = createIndexMappingFromDict(
        node_to_count, additional_keys=node_special_keys
    )

    target_to_index, index_to_target = createIndexMappingFromDict(
        target_to_count, additional_keys=target_special_keys
    )

    return Vocabularies(
        subtoken_to_index=subtoken_to_index,
        index_to_subtoken=index_to_subtoken,
        node_to_index=node_to_index,
        index_to_node=index_to_node,
        target_to_index=target_to_index,
        index_to_target=index_to_target,
    )

