import argparse
import os
import pickle
from utilities.data.Vocabulary import Vocabulary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", required=True)

    args = parser.parse_args()

    outfolder = os.path.dirname(args.i)
    node_dict_file = os.path.join(outfolder, "node_dict.pkl")
    with open(args.i) as f:
        lines = f.readlines()
        node_dict = {
            item.split("|")[0]: i
            for i, item in enumerate(lines)
        }
    
    node_dict[Vocabulary.PAD_TOKEN] = len(node_dict)
    node_dict[Vocabulary.UNK_TOKEN] = len(node_dict)

    with open(node_dict_file, "wb") as f:
        pickle.dump(node_dict, f)

    print(f"Written node dict to: {node_dict_file}")
