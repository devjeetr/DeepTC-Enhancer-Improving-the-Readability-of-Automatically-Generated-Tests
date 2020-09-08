from argparse import ArgumentParser, Namespace

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from models.Transformer.TransformerLightning import TransformerLightning
from models.Transformer.ModelCheckpointer import ModelCheckpointer
from utilities.data.BPEVocabulary import BPEVocabulary
from utilities.data.WordVocabulary import WordVocabulary


def cmd_args():
    parser = ArgumentParser()

    # file paths
    parser.add_argument(
        "--dict-path",
        help="path to dict file containing pickled subtoken, node and target counts",
    )

    # vocabulary settings
    parser.add_argument("--use-bpe", action="store_true")
    parser.add_argument("--subtoken-vocab", help="path to subtoken vocab file")
    parser.add_argument("--subtoken-merges", help="path to subtoken merges")
    parser.add_argument("--target-vocab", help="path to target vocab")
    parser.add_argument("--target-merges", help="path to target merges")
    parser.add_argument("--node-dict", help="path to node dict")
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--val-path", required=True)

    # line cache
    parser.add_argument(
        "--train-line-cache", required=False,
    )
    parser.add_argument(
        "--test-line-cache", required=False,
    )
    parser.add_argument(
        "--val-line-cache", required=False,
    )

    # sequence lengths for path, subtoken and target
    parser.add_argument("--ast-len", default=9, type=int)
    parser.add_argument("--subtoken-len", default=7, type=int)
    parser.add_argument("--target-len", default=4, type=int)
    parser.add_argument("--max-contexts", default=100, type=int)

    # embeddings
    parser.add_argument("--subtoken-embedding-dims", default=128, type=int)
    parser.add_argument("--ast-embedding-dims", default=128, type=int)
    parser.add_argument("--target-embedding-dims", default=128, type=int)

    # some model settings
    parser.add_argument("--context-encoder-dropout", default=0.5, type=float)
    parser.add_argument(
        "--ast-bidirectional",
        default=True,
        help="whether or not to use bidirectional LSTM to encode AST nodes",
    )
    parser.add_argument(
        "--ast-hidden-size", default=256, help="number of hidden units in the AST LSTM"
    )

    # training settings
    parser.add_argument("--shuffle-contexts", default=True, type=bool)
    parser.add_argument("--shuffle-batches", action="store_true")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--predict-variables", action="store_true")

    # data loader settings
    parser.add_argument("--dataloader-num-workers", default=4, type=int)

    # model checkpointing
    parser.add_argument("--save-top-k", default=1, type=int)
    parser.add_argument("--noam-warmup-steps", default=4000, type=int)

    return parser


if __name__ == "__main__":
    parser = cmd_args()

    parser = Trainer.add_argparse_args(parser)
    parser = TransformerLightning.add_model_specific_args(parser)
    hparams = parser.parse_args()
    trainer = Trainer.from_argparse_args(hparams)
    model = TransformerLightning(hparams)

    trainer.fit(model)
