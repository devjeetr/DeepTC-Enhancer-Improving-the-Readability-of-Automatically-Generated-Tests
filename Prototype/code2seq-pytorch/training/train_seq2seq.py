import os
import random

import blessings
import torch
from colorama import Fore, init
from torch import nn
from torch import optim as optim
from torch.utils.data import DataLoader
from apex import amp
from utilities.C2SDataSet import C2SDataSet
from utilities.common import create_vocabularies_from_frequency_dict
from utilities.config import Config, GeneralParams, ModelConfig, SpecialCharacters
from utilities.print_utils import print_model_summary, print_random_training_prediction
from models.seq2seq.Seq2Seq import Seq2Seq
import humanfriendly
import time
import adabound

t = blessings.Terminal()
init(autoreset=True)

data_root = "./data/java-test/"
# data_root = "/media/devjeetroy/ss/backup/Research/TestingProjectV2/tmp/code2seq/data/java-large"
# train_file_path = os.path.join(data_root, "java-test.train.c2s")
train_file_path = "/media/devjeetroy/ss/testing project training projects/code2seq-data/processed/train.c2s"
test_file_path = os.path.join(data_root, "java-test.test.c2s")
# dict_file_path = os.path.join(data_root, "java-test.dict.c2s")
dict_file_path = "/media/devjeetroy/ss/testing project training projects/code2seq-data/processed/java-test.dict.c2s"


special_characters = SpecialCharacters()
token_special_characters = {
    special_characters.subtoken_unk_token,
    special_characters.subtoken_pad_token,
}

node_special_characters = {
    special_characters.node_pad_token,
    special_characters.node_unk_token,
}

target_special_characters = [
    special_characters.target_pad_token,
    special_characters.target_unk_token,
    special_characters.target_eos_token,
    special_characters.target_sos_token,
]


vocabularies = create_vocabularies_from_frequency_dict(
    dict_file_path,
    token_special_characters,
    node_special_characters,
    target_special_characters,
)

model_config = ModelConfig(
    subtoken_embedding_vocab_size=len(vocabularies.subtoken_to_index),
    ast_embedding_vocab_size=len(vocabularies.node_to_index),
    decoder_embedding_vocab_size=len(vocabularies.target_to_index),
)

general_params = GeneralParams(
    batch_size=512,
    max_contexts=100,
    subtoken_length=7,
    ast_path_length=9,
    target_length=4,
)

config = Config(
    vocabularies,
    special_characters=special_characters,
    model_config=model_config,
    general_params=general_params,
)

use_cuda = True
if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

dataset = C2SDataSet.fromConfig(train_file_path, config)
dataloader = DataLoader(
    dataset, batch_size=config.general_params.batch_size, pin_memory=True, shuffle=True
)

n_examples = len(dataloader)

print(f"target_vocab: {len(vocabularies.target_to_index)}")
print(f"node_vocab: {len(vocabularies.node_to_index)}")
print(f"subtoken_vocab: {len(vocabularies.subtoken_to_index)}")

model = Seq2Seq(config, device).to(device)

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
criterion = nn.CrossEntropyLoss(
    ignore_index=config.vocabularies.target_to_index[
        config.special_characters.target_pad_token
    ],
)

# model, optimizer = amp.initialize(
#     model, optimizer, opt_level="O2", keep_batchnorm_fp32=True, loss_scale="dynamic"
# )

print_model_summary(model)
print(model)


def train(model, optimizer, criterion, config, dataloader, current_epoch=-1, mixed_precision=False, grad_clip=5.0):
    start_time = time.time()

    model.train()
    best_loss = float("inf")

    for (
        i,
        (
            label,
            (start, end, path, masks, start_lengths, end_lengths, ast_path_lengths),
        ),
    ) in enumerate(dataloader):
        print()

        if current_epoch != -1:
            epoch_as_str = f" Epoch #{current_epoch} "
            half_len = int(len(epoch_as_str) / 2)
            initial_pad_len = int((t.width / 2 - half_len))
            print(
                "=" * initial_pad_len
                + epoch_as_str
                + "=" * (t.width - initial_pad_len - len(epoch_as_str))
            )
        else:
            print("=" * t.width)

        (start, end, path, masks, start_lengths, end_lengths, ast_path_lengths) = map(
            lambda t: t.to(device),
            (start, end, path, masks, start_lengths, end_lengths, ast_path_lengths),
        )
        label = label.to(device)

        # clear loss gradients
        optimizer.zero_grad()

        # train model
        output = model(
            start,
            end,
            path,
            start_lengths,
            end_lengths,
            ast_path_lengths,
            masks,
            label,
            teacher_force=0.5,
        )

        # print random evaluation
        print_random_training_prediction(
            output, label, config.vocabularies.index_to_target
        )

        # transform output and label for loss calculation
        output = output.view(-1, output.shape[-1])
        label = label.permute(1, 0)
        # trg = label[:, 1:]
        trg = label[1:].reshape(-1)
        # output = output.reshape(-1, output.shape[-1])
        assert trg.shape[0] == output.shape[0]

        # compute loss and backpropogate
        loss = criterion(output, trg)

        if mixed_precision:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 5.0)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            loss.backward()



        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        # apply gradients
        optimizer.step()
        print(
            f"Current loss: {(Fore.GREEN + f'{loss.item()}') if loss.item() < best_loss else loss.item()}"
        )
        best_loss = min(loss.item(), best_loss)
        elapsed = time.time() - start_time

        print(
            f"Time elapsed: {humanfriendly.format_timespan(elapsed)}, throughput: {((i + 1) * config.general_params.batch_size) / elapsed :.02f}"
        )

        print(f"Best loss: {best_loss}")
        print(f"{(i + 1)} / {n_examples}")

        print("=" * t.width)
        print()


def save_checkpoints(destination, model, optimizer, epoch):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    torch.save(checkpoint, destination)


def train_iters(n_epochs=1, checkpoints=None):
    print()
    for epoch in range(n_epochs):
        print(f"Epoch #{epoch}")
        train(model, optimizer, criterion, config, dataloader, current_epoch=epoch)
        print(f"Epoch #{epoch} completed")

        if checkpoints is not None:
            checkpoint_file = os.path.join(checkpoints, f"epoch-0.pt")
            save_checkpoints(checkpoint_file, model, optimizer, epoch)

    return model


# print(config.vocabularies.target_to_index[config.special_characters.target_pad_token])
