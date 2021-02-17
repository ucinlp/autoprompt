import torch


def compute_accuracy(metrics, denom):
    return metrics['accuracy'] / (denom + 1e-13)


def compute_mcc(metrics, denom):
    mcc_numerator = (metrics['TP'] * metrics['TN'] - 
                     metrics['FP'] * metrics['FN'])
    mcc_denominator = torch.sqrt((metrics['TP'] + metrics['FP']) *
                                 (metrics['TP'] + metrics['FN']) *
                                 (metrics['TN'] + metrics['FP']) *
                                 (metrics['TN'] + metrics['FN']))
    return mcc_numerator / mcc_denominator


def compute_f1(metrics, denom):
    precision = metrics['TP'] / (metrics['TP'] + metrics['FP'])
    recall = metrics['TP'] / (metrics['TP'] + metrics['FN'])
    return (2 * precision * recall) / (precision + recall)


METRICS = {
    'accuracy': compute_accuracy,
    'MCC': compute_mcc,
    'F1': compute_f1,
}

