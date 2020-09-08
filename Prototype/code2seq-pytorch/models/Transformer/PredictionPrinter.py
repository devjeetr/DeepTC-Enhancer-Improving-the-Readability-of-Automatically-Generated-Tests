import pytorch_lightning as pl

class PredictionPrinter(pl.Callback):
    def on_train_end(self, trainer, model):
        print("HEE")