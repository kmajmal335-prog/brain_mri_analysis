import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()

        intersection = (pred * target).sum(dim=2).sum(dim=2)
        dice = (2. * intersection + self.smooth) / (
                pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + self.smooth
        )

        return 1 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """Combined loss for segmentation: Dice + BCE + Focal"""

    def __init__(self, dice_weight=0.5, bce_weight=0.3, focal_weight=0.2):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()
        self.focal_loss = FocalLoss()

        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)

        # Convert to class format for focal loss
        pred_class = (pred > 0.5).float()
        target_class = target.long().squeeze(1)

        # Create 2-class problem for focal loss
        pred_focal = torch.stack([1 - pred.squeeze(1), pred.squeeze(1)], dim=1)
        focal = self.focal_loss(pred_focal, target_class)

        total_loss = (self.dice_weight * dice +
                      self.bce_weight * bce +
                      self.focal_weight * focal)

        return total_loss, {'dice': dice, 'bce': bce, 'focal': focal}