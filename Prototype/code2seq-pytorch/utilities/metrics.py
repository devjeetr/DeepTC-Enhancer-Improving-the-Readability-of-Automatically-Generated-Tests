from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import precision_score, accuracy_score, roc_auc_score, recall_score


def compute_metric(labels, predictions, ignore_idx, scorer, *args, **kwargs):
    y, y_hat = [], []

    for label, prediction in zip(labels, predictions):
        try:
            max_len = label.index(ignore_idx)
        except:
            max_len = len(label)

        trimmed_label = label[:max_len]
        trimmed_prediction = prediction[:max_len]

        y.extend(trimmed_label)
        y_hat.extend(trimmed_prediction)

    assert len(y) == len(y_hat)

    return scorer(y, y_hat, *args, **kwargs)


def compute_precision(labels, predictions, ignore_idx):
    return compute_metric(labels, predictions, ignore_idx, precision_score, average="micro")

def compute_accuracy(labels, predictions, ignore_idx):
    return compute_metric(labels, predictions, ignore_idx, accuracy_score)

def compute_recall(labels, predictions, ignore_idx):
    return compute_metric(labels, predictions, ignore_idx, recall_score, average="micro")
