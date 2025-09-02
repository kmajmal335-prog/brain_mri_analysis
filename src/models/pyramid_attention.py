import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from efficientnet_pytorch import EfficientNet


class PyramidAttentionModule(nn.Module):
    """Multi-scale Pyramid Attention Module"""

    def __init__(self, in_channels, reduction=16):
        super(PyramidAttentionModule, self).__init__()

        # Multi-scale branches
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 3, padding=1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True)
        )

        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 5, padding=2),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True)
        )

        self.branch_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True)
        )

        # Attention fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels // reduction * 4, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()

        # Multi-scale features
        feat1 = self.branch1(x)
        feat3 = self.branch3(x)
        feat5 = self.branch5(x)
        feat_pool = F.interpolate(self.branch_pool(x), size=(h, w), mode='bilinear', align_corners=False)

        # Concatenate multi-scale features
        multi_scale = torch.cat([feat1, feat3, feat5, feat_pool], dim=1)

        # Generate attention map
        attention = self.fusion(multi_scale)

        # Apply attention
        out = x * attention + x

        return out, attention


class SpatialPyramidPooling(nn.Module):
    """Spatial Pyramid Pooling Module"""

    def __init__(self, in_channels, pool_sizes=[1, 2, 4, 8]):
        super(SpatialPyramidPooling, self).__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels

        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(pool_size) for pool_size in pool_sizes
        ])

        # Calculate total output channels
        total_channels = in_channels * sum([p * p for p in pool_sizes])

        self.conv = nn.Sequential(
            nn.Conv2d(total_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        batch_size, c, h, w = x.size()

        features = []
        for pool in self.pools:
            feat = pool(x)
            feat = feat.view(batch_size, c, -1)
            features.append(feat)

        # Concatenate all pooled features
        features = torch.cat(features, dim=2)
        features = features.view(batch_size, -1, 1, 1)

        # Project back to original channels
        out = self.conv(features)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)

        return out


class EnhancedPyramidClassifier(nn.Module):
    """Enhanced classifier with Pyramid Attention"""

    def __init__(self, num_classes=4, backbone='efficientnet-b4'):
        super(EnhancedPyramidClassifier, self).__init__()

        # Backbone
        if 'efficientnet' in backbone:
            self.backbone = EfficientNet.from_pretrained(backbone)
            num_features = self.backbone._fc.in_features
            self.backbone._fc = nn.Identity()
        else:
            self.backbone = models.resnet50(pretrained=True)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        # Pyramid Attention Modules at different scales
        self.pam1 = PyramidAttentionModule(num_features // 4)
        self.pam2 = PyramidAttentionModule(num_features // 2)
        self.pam3 = PyramidAttentionModule(num_features)

        # Spatial Pyramid Pooling
        self.spp = SpatialPyramidPooling(num_features)

        # Enhanced classifier head with dropout and batch norm
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

        # Auxiliary classifier for deep supervision
        self.aux_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Extract features
        features = self.backbone.extract_features(x) if hasattr(self.backbone, 'extract_features') else self.backbone(x)

        # Apply pyramid attention at different scales
        if len(features.shape) == 4:  # Conv features
            features, att = self.pam3(features)

            # Spatial pyramid pooling
            spp_features = self.spp(features)

            # Global average pooling
            gap_features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
            gmp_features = F.adaptive_max_pool2d(features, 1).squeeze(-1).squeeze(-1)

            # Combine features
            combined_features = torch.cat([gap_features, gmp_features], dim=1)
        else:
            combined_features = torch.cat([features, features], dim=1)
            att = None

        # Main classification
        main_out = self.classifier(combined_features)

        # Auxiliary classification
        aux_out = self.aux_classifier(gap_features if len(features.shape) == 4 else features)

        return main_out, aux_out, att