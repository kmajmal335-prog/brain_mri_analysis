import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet


class BrainTumorClassifier(nn.Module):
    def __init__(self, num_classes=4, backbone='efficientnet-b3', dropout_rate=0.5):
        super(BrainTumorClassifier, self).__init__()

        self.backbone = backbone
        self.num_classes = num_classes

        if 'efficientnet' in backbone:
            self.base_model = EfficientNet.from_pretrained(backbone)
            num_features = self.base_model._fc.in_features
            self.base_model._fc = nn.Identity()
        elif backbone == 'resnet50':
            self.base_model = models.resnet50(pretrained=True)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        elif backbone == 'densenet121':
            self.base_model = models.densenet121(pretrained=True)
            num_features = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Identity()

        # Custom head
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_classes)
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(num_features, num_features // 8),
            nn.ReLU(),
            nn.Linear(num_features // 8, num_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.base_model(x)

        # Apply attention
        attention_weights = self.attention(features)
        features = features * attention_weights

        # Classification
        output = self.head(features)

        return output, attention_weights

    def get_features(self, x):
        """Extract features for visualization"""
        return self.base_model(x)