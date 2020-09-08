from pytorch_lightning import Trainer
from models.Transformer.TransformerLightning import TransformerLightning
import torch
from utilities import SpecialCharacters
from argparse import Namespace
from utilities.print_utils import pp_prediction

test_line_cache = "/media/devjeetroy/ss/testing project training projects/code2seq-data/processed/test-line-cache.pkl"
test_path = "/media/devjeetroy/ss/testing project training projects/code2seq-data/processed/test.c2s"

ckp = "./lightning_logs/version_15/checkpoints/epoch=12.ckpt"
# model = TransformerLightning.load_from_checkpoint(ckp)

checkpoint = torch.load(ckp)
hparams = Namespace(**checkpoint["hparams"])
hparams.test_line_cache = test_line_cache
hparams.test_path = test_path

# print(hparams.keys())

model = TransformerLightning(hparams).cuda()
model.load_state_dict(checkpoint["state_dict"])
index_to_target = {v: k for k, v in hparams.target_to_index.items()}
pad_idx = hparams.target_to_index[SpecialCharacters.PAD_TOKEN]

print("Starting predictions")
for batch in model.test_dataloader():
    print(f"making single prediction")

    (
        labels,
        (start, end, path, masks, start_lengths, end_lengths, ast_path_lengths),
    ) = batch
    labels = labels.cuda()
    start = start.cuda()
    end = end.cuda()
    path = path.cuda()
    masks = masks.cuda()
    start_lengths = start_lengths.cuda()
    end_lengths = end_lengths.cuda()
    ast_path_lengths = ast_path_lengths.cuda()
    batch = (
        labels,
        (start, end, path, masks, start_lengths, end_lengths, ast_path_lengths),
    )

    predictions, raw = model.predict(batch)
    print(f"predictions complete")
    predictions = predictions.tolist()
    labels = labels.tolist()

    metrics = model.get_metrics(labels, predictions, pad_idx)

    prediction = [index_to_target[k] for k in predictions[0]]

    label = [index_to_target[k] for k in labels[0]]
    # print(predictions[0])
    # print(label[0])

    pp_prediction(prediction, label,)
    print(metrics)