"""Evaluation metrics."""

from collections import defaultdict
import math
from typing import List

import torch


class Metric:
    score_key = None
    keys = None

    def __init__(self, label_map):
        self.label_map = label_map
        self.reset()

    def reset(self):
        raise NotImplementedError

    def update(self, labels, predictions):
        raise NotImplementedError

    def reduce(self):
        raise NotImplementedError

    def get(self):
        raise NotImplementedError


class Accuracy(Metric):
    score_key = 'accuracy'
    keys = ['accuracy']

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, labels, predictions):
        self.correct += labels.eq(predictions).sum()
        self.total += labels.size(0)

    def reduce(self):
        torch.distributed.reduce(self.correct, 0)
        torch.distributed.reduce(self.total, 0)

    def get(self):
        return {'accuracy': self.correct.item() / (self.total + 1e-13)}


class BinaryF1(Metric):
    score_key = 'f1'
    keys = ['precision', 'recall', 'f1']

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, labels, predictions):
        self.tp += torch.sum(labels.eq(1) & predictions.eq(1))
        self.fp += torch.sum(labels.eq(0) & predictions.eq(1))
        self.fn += torch.sum(labels.eq(1) & predictions.eq(0))

    def reduce(self):
        torch.distributed.reduce(self.tp, 0)
        torch.distributed.reduce(self.fp, 0)
        torch.distributed.reduce(self.fn, 0)

    def get(self):
        precision = self.tp / (self.tp + self.fp)
        recall = self.tp / (self.tp + self.fn)
        f1 = 2 * precision * recall / (precision + recall)
        return {
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item(),
        }


class AccuracyF1(Metric):
    score_key = 'f1'
    keys = [*Accuracy.keys, *BinaryF1.keys]

    def __init__(self, label_map):
        self.label_map = label_map  # TODO(rloganiv): IDK
        self.accuracy = Accuracy(label_map)
        self.f1 = BinaryF1(label_map)

    def reset(self):
        self.accuracy.reset()
        self.f1.reset()

    def update(self, labels, predictions):
        self.accuracy.update(labels, predictions)
        self.f1.update(labels, predictions)

    def reduce(self):
        self.accuracy.reduce()
        self.f1.reduce()

    def get(self):
        return {**self.accuracy.get(), **self.f1.get()}


class MacroF1(Metric):
    score_key = 'f1'
    keys = ['precision', 'recall', 'f1']

    def reset(self):
        self.tp = torch.zeros(len(self.label_map))
        self.fp = torch.zeros(len(self.label_map))
        self.fn = torch.zeros(len(self.label_map))

    def update(self, labels, predictions):
        # TODO(rloganiv): This is kind of hacky, but idk if there's a better
        # way.
        self.tp = self.tp.to(labels.device)
        self.fp = self.fp.to(labels.device)
        self.fn = self.fn.to(labels.device)

        for i in range(len(self.label_map)):
            self.tp[i] += torch.sum(labels.eq(i) & predictions.eq(i))
            self.fp[i] += torch.sum(labels.ne(i) & predictions.eq(i))
            self.fn[i] += torch.sum(labels.eq(i) & predictions.ne(i))

    def reduce(self):
        torch.distributed.reduce(self.tp, 0)
        torch.distributed.reduce(self.fp, 0)
        torch.distributed.reduce(self.fn, 0)

    def get(self):
        precision = self.tp / (self.tp + self.fp + 1e-13)
        recall = self.tp / (self.tp + self.fn + 1e-13)
        f1 = 2 * precision * recall / (precision + recall + 1e-13)
        return {
            'precision': precision.mean().item(),
            'recall': recall.mean().item(),
            'f1': f1.mean().item(),
        }


class MatthewsCorrelation(Metric):
    score_key = 'matthews_correlation'
    keys = ['matthews_correlation']

    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def update(self, labels, predictions):
        self.tp += torch.sum(labels.eq(1) & predictions.eq(1)).item()
        self.tn += torch.sum(labels.eq(0) * predictions.eq(0)).item()
        self.fp += torch.sum(labels.eq(0) & predictions.eq(1)).item()
        self.fn += torch.sum(labels.eq(1) & predictions.eq(0)).item()

    def reduce(self):
        torch.distributed.reduce(self.tp, 0)
        torch.distributed.reduce(self.tn, 0)
        torch.distributed.reduce(self.fp, 0)
        torch.distributed.reduce(self.fn, 0)


    def get(self):
        numerator = self.tp * self.tn - self.fp * self.fn
        denominator = math.sqrt(
            (self.tp + self.fp) *
            (self.tp + self.fn) * 
            (self.tn + self.fp) *
            (self.tn + self.fn)
        )
        return {'matthews_correlation': numerator / (denominator + 1e-13)}


METRICS = {
    'accuracy': Accuracy,
    'accuracy-f1': AccuracyF1,
    'binary-f1': BinaryF1,
    'macro-f1': MacroF1,
    'matthews-correlation': MatthewsCorrelation,
}
