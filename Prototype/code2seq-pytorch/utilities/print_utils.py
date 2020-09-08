from millify import millify
from colorama import Fore, init
import random
import torch
import blessings

init(autoreset=True)

t = blessings.Terminal()


def get_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model):
    n_trainable_model = get_trainable_parameters(model)
    print(f"Initialized encoder with {millify(n_trainable_model)} trainable parameters")


def colorize_loss(loss):
    if loss > 6:
        return f"{Fore.RED}{loss}"
    elif loss > 5:
        return f"{Fore.YELLOW}{loss}"
    elif loss > 4:
        return f"{Fore.BLUE}{loss}"
    else:
        return f"{Fore.GREEN}{loss}"


def pp_prediction(prediction, label):
    prediction_output = []
    label_output = []

    for (pred_token, label_token) in zip(prediction, label):
        token_len = max(len(pred_token), len(label_token))
        if pred_token == label_token:
            prediction_output.append(f"{t.bold_green(pred_token.rjust(token_len))}")
            label_output.append(f"{t.bold_green(label_token.rjust(token_len))}")
        else:
            prediction_output.append(f"{t.red(pred_token.rjust(token_len))}")
            label_output.append(f"{t.red(label_token.rjust(token_len))}")

    print(f"label:      {'|'.join(label_output)}")
    print(f"prediction: {'|'.join(prediction_output)}")


def print_random_training_prediction(
    output: torch.Tensor, label: torch.Tensor, vocabulary
):
    rand_index = random.randint(0, label.shape[0] - 1)

    sample_target = label[rand_index, :]
    sample_pred = output[rand_index, :, :]
    sample_pred = sample_pred.argmax(1)
    sample_pred = sample_pred.tolist()
    sample_target = sample_target.tolist()
    sample_target = vocabulary.decode_target(sample_target)
    sample_pred = vocabulary.decode_target(sample_pred)

    print(f"target: {sample_target}")
    print(f"Prediction: {sample_pred}")
    # pp_prediction(sample_pred, sample_target[1:])


def print_topk(
    output: torch.Tensor, label: torch.Tensor, index_to_target: dict
):
    rand_index = random.randint(0, label.shape[0] - 1)

    sample_target = label[rand_index, :]
    sample_pred = output[rand_index, :, :]
    print(f"sample pred shape: {sample_pred.shape}")
    sample_pred = sample_pred.topk(2, dim=0)

    print(sample_pred.indices.shape)
    sample_pred = sample_pred.indices.tolist()
    # print(f"Top K: {sample_pred}")
    # sample_target = sample_target.tolist()
    # sample_target = [index_to_target[x] for x in sample_target]
    # sample_pred = [index_to_target[x] for x in sample_pred]

    # pp_prediction(sample_pred, sample_target[1:])
