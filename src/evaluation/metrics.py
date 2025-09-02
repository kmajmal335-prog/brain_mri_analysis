import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import torch


class MetricsCalculator:
    def __init__(self, num_classes=4):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.predictions = []
        self.targets = []
        self.probabilities = []

    def update(self, preds, targets, probs=None):
        self.predictions.extend(preds)
        self.targets.extend(targets)
        if probs is not None:
            self.probabilities.extend(probs)

    def compute_classification_metrics(self):
        metrics = {
            'accuracy': accuracy_score(self.targets, self.predictions),
            'precision': precision_score(
                self.targets, self.predictions, average='weighted'
            ),
            'recall': recall_score(
                self.targets, self.predictions, average='weighted'
            ),
            'f1': f1_score(
                self.targets, self.predictions, average='weighted'
            )
        }

        if self.probabilities:
            metrics['auroc'] = roc_auc_score(
                self.targets,
                self.probabilities,
                multi_class='ovr',
                average='weighted'
            )

        metrics['confusion_matrix'] = confusion_matrix(
            self.targets, self.predictions
        )

        metrics['classification_report'] = classification_report(
            self.targets, self.predictions
        )

        return metrics

    @staticmethod
    def dice_coefficient(pred, target, smooth=1e-6):
        pred = pred.flatten()
        target = target.flatten()
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return dice

    @staticmethod
    def iou_score(pred, target, smooth=1e-6):
        pred = pred.flatten()
        target = target.flatten()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou

    @staticmethod
    def sensitivity(pred, target):
        true_positives = (pred * target).sum()
        actual_positives = target.sum()
        return true_positives / (actual_positives + 1e-6)

    @staticmethod
    def specificity(pred, target):
        true_negatives = ((1 - pred) * (1 - target)).sum()
        actual_negatives = (1 - target).sum()
        return true_negatives / (actual_negatives + 1e-6)