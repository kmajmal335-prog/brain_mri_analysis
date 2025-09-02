import torch
import torch.nn as nn
from .classifier import BrainTumorClassifier
from .segmenter import AttentionUNet


class HybridBrainTumorModel(nn.Module):
    """Joint classification and segmentation model"""

    def __init__(self, num_classes=4, backbone='efficientnet-b3'):
        super(HybridBrainTumorModel, self).__init__()

        # Classification branch
        self.classifier = BrainTumorClassifier(
            num_classes=num_classes,
            backbone=backbone
        )

        # Segmentation branch
        self.segmenter = AttentionUNet(in_channels=3, out_channels=1)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        # Combined prediction head
        self.combined_head = nn.Linear(128 + num_classes, num_classes)

    def forward(self, x):
        # Get classification output
        clf_output, attention = self.classifier(x)

        # Get segmentation output
        seg_output = self.segmenter(x)

        # Fuse information
        seg_features = self.fusion(seg_output)

        # Combine features for refined classification
        combined = torch.cat([seg_features, clf_output], dim=1)
        refined_clf = self.combined_head(combined)

        return {
            'classification': refined_clf,
            'segmentation': seg_output,
            'attention': attention,
            'initial_clf': clf_output
        }