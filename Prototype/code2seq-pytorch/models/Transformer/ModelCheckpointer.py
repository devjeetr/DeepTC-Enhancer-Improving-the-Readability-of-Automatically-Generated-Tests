import pytorch_lightning as pl
import torch
import os
from argparse import Namespace
import pickle

def clean_namespace(hparams):
    """
    Removes all functions from hparams so we can pickle
    :param hparams:
    :return:
    """

    if isinstance(hparams, Namespace):
        del_attrs = []
        for k in hparams.__dict__:
            if callable(getattr(hparams, k)):
                del_attrs.append(k)

        for k in del_attrs:
            delattr(hparams, k)

    elif isinstance(hparams, dict):
        del_attrs = []
        for k, v in hparams.items():
            if callable(v):
                del_attrs.append(k)

        for k in del_attrs:
            del hparams[k]

class ModelCheckpointer(pl.Callback):
    def __init__(self, folder="state_dicts"):
        super().__init__()
        self.folder = folder

    def on_epoch_end(self, trainer: pl.Trainer, model: pl.LightningModule):
        save_root = os.path.join(trainer.ckpt_path, self.folder)
        if not os.path.exists(save_root):
            os.makedirs(save_root)

        filename = f"state_dict.epoch={trainer.current_epoch}.ckpt"

        with open(os.path.join(save_root, filename), "wb") as f:
            pickle.dump(
                model.state_dict(), f
            )
