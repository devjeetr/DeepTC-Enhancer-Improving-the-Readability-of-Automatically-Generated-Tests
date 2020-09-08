from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.Transformer.CustomTransformer import CustomTransformer
from models.Transformer.optimizer import NoamOptimizer
from utilities import metrics
from utilities.C2SDataSet import C2SDataSet
from utilities.data.BPEVocabulary import BPEVocabulary
from utilities.data.Vocabulary import Vocabulary
from utilities.data.WordVocabulary import WordVocabulary
from utilities.print_utils import print_random_training_prediction


def create_transformer_from_hparams(
    hparams, vocabulary: Vocabulary
) -> CustomTransformer:

    return CustomTransformer(
        max_contexts=hparams.max_contexts,
        subtoken_len=hparams.subtoken_len,
        subtoken_embedding_dims=hparams.subtoken_embedding_dims,
        subtoken_vocab=vocabulary.terminal_vocab_size(),
        subtoken_pad_idx=vocabulary.encode_terminal(vocabulary.PAD_TOKEN),
        ast_path_len=hparams.ast_len,
        ast_embedding_dims=hparams.ast_embedding_dims,
        ast_vocab=vocabulary.node_vocab_size(),
        ast_hidden_size=hparams.ast_hidden_size,
        ast_pad_idx=vocabulary.encode_node(vocabulary.PAD_TOKEN),
        label_vocab=vocabulary.target_vocab_size(),
        label_pad_idx=vocabulary.encode_target(vocabulary.PAD_TOKEN),
        bidirectional=hparams.ast_bidirectional,
        context_embedding_dim=hparams.d_model,
        d_model=hparams.d_model,
        nhead=hparams.nhead,
        num_encoder_layers=hparams.num_encoder_layers,
        num_decoder_layers=hparams.num_decoder_layers,
        transformer_feedforward=hparams.transformer_feedforward,
        transformer_dropout=hparams.transformer_dropout,
        transformer_activation=hparams.transformer_activation,
        context_encoder_dropout=hparams.context_encoder_dropout,
    )


class TransformerLightning(pl.LightningModule):
    def __init__(self, hparams, *args, **kwargs):
        super(TransformerLightning, self).__init__()
        self.hparams = hparams

        if not hparams.use_bpe:
            assert (
                hparams.dict_path is not None
            ), "If not using BPE, --dict-path must be set"
            self.vocabulary = WordVocabulary(hparams)
        else:

            self.vocabulary = BPEVocabulary(hparams)
            if hparams.predict_variables:

                for token in [
                    "METHOD_NAME",
                    "TARGET_VARIABLE",
                    "VARIABLE",
                    Vocabulary.PAD_TOKEN,
                ]:
                    self.vocabulary.add_special_terminal_token(token)
            else:
                for token in ["METHOD_NAME", "VARIABLE", Vocabulary.PAD_TOKEN]:
                    self.vocabulary.add_special_terminal_token(token)

            for token in [
                Vocabulary.PAD_TOKEN,
                Vocabulary.SOS_TOKEN,
                Vocabulary.EOS_TOKEN,
                Vocabulary.UNK_TOKEN,
            ]:
                self.vocabulary.add_special_target_token(token)

        self.c2s_transformer = create_transformer_from_hparams(hparams, self.vocabulary)

    def forward(self, *args, **kwargs):
        return self.c2s_transformer(*args, **kwargs)

    def compute_batch_loss(self, label, prediction):
        prediction = prediction.reshape(-1, prediction.shape[-1])
        trg = label[:, 1:].reshape(-1)

        assert trg.shape[0] == prediction.shape[0]
        loss = F.cross_entropy(
            prediction,
            trg,
            ignore_index=self.vocabulary.encode_target(self.vocabulary.PAD_TOKEN),
        )

        return loss

    def training_step(self, batch, batch_idx):
        (
            label,
            (start, end, path, masks, start_lengths, end_lengths, ast_path_lengths),
        ) = batch

        prediction = self(
            start,
            end,
            path,
            start_lengths,
            end_lengths,
            ast_path_lengths,
            masks,
            label[:, :-1],
        )

        # TODO
        # Remove
        print(f"encoding TARGET_VARIABLE")
        print(self.vocabulary.encode_terminal(self.vocabulary.PAD_TOKEN))
        # print_random_training_prediction(prediction, label, self.vocabulary)
        # End

        loss = self.compute_batch_loss(label, prediction)
        return {"loss": loss, "log": {"loss": loss}}

    @staticmethod
    def raw_pred_to_indices(prediction: torch.Tensor):
        return F.softmax(prediction, dim=2).argmax(2)

    def predict(self, batch):
        (
            label,
            (start, end, path, masks, start_lengths, end_lengths, ast_path_lengths),
        ) = batch

        label = label
        model = self.c2s_transformer
        batch_size = label.shape[0]

        # compute encoder output
        with torch.no_grad():
            trg_raw = None
            trg_output = (
                torch.ones(batch_size, 1).to(label.device)
                * self.vocabulary.encode_target(self.vocabulary.SOS_TOKEN)
            ).long()

        for i in range(0, self.hparams.target_len + 1):
            with torch.no_grad():
                output = model(
                    start,
                    end,
                    path,
                    start_lengths,
                    end_lengths,
                    ast_path_lengths,
                    masks,
                    trg_output,
                )

                # apply softmax and argmax to get the latest predicted token
                predictions = self.raw_pred_to_indices(output)
                trg_output = torch.cat(
                    [trg_output, predictions[:, -1].unsqueeze(1)], dim=1
                )
                # print(f"new shape: {trg_raw.shape}")

                if trg_raw is None:
                    trg_raw = output.clone()
                else:
                    trg_raw = torch.cat([trg_raw, output[:, -1, :].unsqueeze(1)], dim=1)

        to_test = self.raw_pred_to_indices(trg_raw)
        assert (to_test == trg_output[:, 1:]).all()

        return trg_output, trg_raw

    @staticmethod
    def get_metrics(y, y_hat, ignore_idx):
        precision = metrics.compute_precision(y, y_hat, ignore_idx)
        recall = metrics.compute_recall(y, y_hat, ignore_idx)
        accuracy = metrics.compute_accuracy(y, y_hat, ignore_idx)

        return {"precision": precision, "recall": recall, "accuracy": accuracy}

    @staticmethod
    def aggregate_metrics(outputs, prefix=""):
        metric_keys = outputs[0].keys()
        epoch_metrics = {}

        for k in metric_keys:
            epoch_metrics[f"{prefix}_{k}"] = torch.tensor(
                [entry[k] for entry in outputs]
            ).mean()

        return {
            f"{prefix}_loss": epoch_metrics[f"{prefix}_loss"],
            "log": {**epoch_metrics},
        }

    def run_evaluation(self, batch, batch_idx):
        (label, _) = batch

        predicted_labels, predicted_raw = self.predict(batch)
        loss = self.compute_batch_loss(label, predicted_raw)
        other_metrics = self.get_metrics(
            label.tolist(),
            predicted_labels.tolist(),
            self.vocabulary.encode_target(self.vocabulary.PAD_TOKEN),
        )

        return {"loss": loss.item(), **other_metrics}

    def validation_step(self, batch, batch_idx):
        return self.run_evaluation(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        return self.aggregate_metrics(outputs, prefix="val")

    def test_step(self, batch, batch_idx):
        return self.run_evaluation(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.aggregate_metrics(outputs, prefix="test")

    def train_dataloader(self,):
        return dataloader_from_hparams(
            self.hparams.train_path,
            self.hparams.train_line_cache,
            self.hparams,
            self.vocabulary,
            shuffle=True,
        )

    def val_dataloader(self,):
        return dataloader_from_hparams(
            self.hparams.val_path,
            self.hparams.val_line_cache,
            self.hparams,
            self.vocabulary,
        )

    def test_dataloader(self,):
        return dataloader_from_hparams(
            self.hparams.test_path,
            self.hparams.test_line_cache,
            self.hparams,
            self.vocabulary,
        )

    def configure_optimizers(self,):
        optimizer = NoamOptimizer(
            self.parameters(),
            self.hparams.d_model,
            warmup_steps=self.hparams.noam_warmup_steps,
        )

        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--d_model", type=int, default=512)
        parser.add_argument(
            "--nhead", type=int, default=8, help="number of heads for multiattention"
        )
        parser.add_argument("--num_encoder_layers", type=int, default=3)
        parser.add_argument("--num_decoder_layers", type=int, default=3)
        parser.add_argument("--transformer-feedforward", type=int, default=512)
        parser.add_argument("--transformer-dropout", type=float, default=0.1)
        parser.add_argument("--transformer-activation", type=str, default="relu")

        return parser


def dataloader_from_hparams(
    path: str,
    line_cache: str,
    hparams: Namespace,
    vocabulary: Vocabulary,
    shuffle=False,
) -> C2SDataSet:
    dataset = C2SDataSet(
        path,
        hparams.max_contexts,
        vocabulary,
        hparams.subtoken_len,
        hparams.ast_len,
        hparams.target_len,
        shuffle=hparams.shuffle_contexts,
        variable_only_filter=hparams.predict_variables,
        line_cache=line_cache,
    )

    return DataLoader(
        dataset,
        batch_size=hparams.batch_size,
        shuffle=shuffle,
        num_workers=hparams.dataloader_num_workers,
    )
