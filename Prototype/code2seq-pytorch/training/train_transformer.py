import datetime
import os

import torch
import torch.nn.functional as F
from blessings import Terminal
from einops import rearrange
from torch import nn
from torch import optim as optim
from torch.utils.data import DataLoader

from utilities.C2SDataSet import C2SDataSet
from utilities.common import create_vocabularies_from_frequency_dict
from utilities.config import Config, GeneralParams, ModelConfig, SpecialCharacters
from utilities.print_utils import colorize_loss, print_model_summary

# from seq2seq.Seq2Seq import Seq2Seq, decoder_from_config, encoder_from_config
from models.Transformer.CustomTransformer import CustomTransformer
from models.Transformer.optimizer import NoamOptimizer, LabelSmoothingLoss

t = Terminal()
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
    decoder_hidden_dim=256,
    encoder_hidden_dim=256,
    decoder_dropout=0.5,
    encoder_dropout=0.5,
)

general_params = GeneralParams(batch_size=128, target_length=4, max_contexts=100)
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
n_examples = len(dataset)


# print("Transferring models to GPU memory")


model = CustomTransformer(config, device).to(device)


print_model_summary(model)
print(model)


def initialize_weights(m):
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


initialize_weights(model)
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
# optimizer = optim.Adam(model.parameters(), lr=0.0005)
optimizer = NoamOptimizer(
    model.parameters(), config.model_config.decoder_hidden_dim, warmup_steps=4000
)

# criterion = LabelSmoothingLoss(
#     0.4,
#     len(config.vocabularies.target_to_index),
#     ignore_index=config.vocabularies.target_to_index[
#         config.special_characters.target_pad_token
#     ],
# )

# criterion.to(device)

criterion = nn.CrossEntropyLoss(
    ignore_index=config.vocabularies.target_to_index[
        config.special_characters.target_pad_token
    ],
)

# criterion = LabelSmoothingLoss


def prettyPrintPrediction(target, prediction):
    pass


def inferSentence(model, example, criterion, config, device):
    (
        label,
        (start, end, path, masks, start_lengths, end_lengths, ast_path_lengths),
    ) = example

    label = label[0]

    with torch.no_grad():
        embedding = model.path_encoder(
            start[0].unsqueeze(0),
            end[0].unsqueeze(0),
            path[0].unsqueeze(0),
            start_lengths[0].unsqueeze(0),
            end_lengths[0].unsqueeze(0),
            ast_path_lengths[0].unsqueeze(0),
        )

        encoder_output = model.encoder(embedding, masks[0].unsqueeze(0).unsqueeze(1).unsqueeze(2),)

    trg_indices = [
        config.vocabularies.target_to_index[config.special_characters.target_sos_token]
    ]

    for i in range(0, config.general_params.target_length + 1):
        trg = torch.LongTensor(trg_indices).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg)
        with torch.no_grad():
            output = model.decoder(
                trg,
                encoder_output,
                trg_mask,
                masks[0].unsqueeze(0).unsqueeze(1).unsqueeze(2),
            )
            out = output
            pred_token = out.argmax(2).flatten()[-1].item()

            trg_indices.append(pred_token)

    target_words = [config.vocabularies.index_to_target[i] for i in label.tolist()]
    actual_words = [config.vocabularies.index_to_target[i] for i in trg_indices]

    target = "|".join(target_words)
    actual = "|".join(actual_words)

    print()
    print(f"target: {target}")
    print(f"actual: {actual}")
    print()


def train(model, optimizer, criterion, config, dataloader):
    for (i, example,) in enumerate(dataloader):
        # model.train()
        optimizer.zero_grad()

        (
            label,
            (start, end, path, masks, start_lengths, end_lengths, ast_path_lengths),
        ) = example

        (start, end, path, masks, start_lengths, end_lengths, ast_path_lengths) = map(
            lambda t: t.to(device),
            (start, end, path, masks, start_lengths, end_lengths, ast_path_lengths),
        )
        label = label.to(device)
        output = model(
            start,
            end,
            path,
            start_lengths,
            end_lengths,
            ast_path_lengths,
            masks,
            label[:, :-1],
        )

        prediction_example = (
            label,
            (start, end, path, masks, start_lengths, end_lengths, ast_path_lengths),
        )
        # print(f"original output shape: {output.shape}")
        # print(f"label shape: {label.shape}")
        rand_index = 0
        sample_target = label[rand_index, :]
        sample_pred = output[rand_index, :, :]
        sample_pred = sample_pred.argmax(1)
        sample_pred = sample_pred.tolist()
        sample_pred_raw = sample_pred
        sample_target = sample_target.tolist()
        sample_target = [config.vocabularies.index_to_target[x] for x in sample_target]
        sample_pred = [config.vocabularies.index_to_target[x] for x in sample_pred]

        output = output.reshape(-1, output.shape[-1])
        trg = label[:, 1:].reshape(-1)
        # trg = label
        # output = output.reshape(-1, output.shape[-1])
        assert trg.shape[0] == output.shape[0]
        # print(trg[0])
        # print(output[0])
        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

        optimizer.step()
        print()
        print(f"{t.bold}Current loss: {colorize_loss(loss.item())}")
        print(f"\n{i+1} / {len(dataloader)}\n\n")
        inferSentence(model, prediction_example, criterion, config, device)

        print(f"target: {sample_target[0:]}")
        print(f"pred:   {sample_pred[0:]}")
        # if i > 1000:
        #     break


model.train()
n_epochs = 10
for epoch in range(n_epochs):
    print(f"Epoch #{epoch}")
    train(model, optimizer, criterion, config, dataloader)
    print(f"Epoch #{epoch} completed")


curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
with open("trained_models/transformer_{curr_time}.pkl", "wb") as f:
    torch.save(model.state_dict(), f)

# print(config.vocabularies.target_to_index[config.special_characters.target_pad_token])
