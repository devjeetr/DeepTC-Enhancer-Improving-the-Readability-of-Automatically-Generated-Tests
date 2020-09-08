from training.train_seq2seq import train_iters
import torch
import argparse
import os
import shutil

def cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", help="Save location for the model")
    parser.add_argument("--model-name", help="model name")
    parser.add_argument("--epochs", help="Number of epochs", default=50)
    return parser.parse_args()


args = cmd_args()


if os.path.exists(args.save):
    print(f"The file {args.save} already exists. Confirm deletion (y / n): ", end="")
    resp = input()
    if resp.lower().startswith("y"):
        shutil.rmtree(args.save)
    else:
        exit(0)

# if not os.path.exists(args.save):
os.makedirs(args.save)

check_point_location = os.path.join(args.save, "checkpoints")
os.makedirs(check_point_location)
model_save_location = os.path.join(args.save, args.model_name)
model = train_iters(n_epochs=int(args.epochs), checkpoints=check_point_location)

print()
print("Saving model..")

torch.save(model, model_save_location)
print(f"Saved model to {model_save_location}")
