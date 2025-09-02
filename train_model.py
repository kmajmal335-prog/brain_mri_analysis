import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import os
from datetime import datetime
import warnings
import random
from collections import Counter
import gc

# Check PyTorch version for compatibility
PYTORCH_VERSION = torch.__version__
print(f"PyTorch Version: {PYTORCH_VERSION}")

# Check if CUDA is available
if torch.cuda.is_available():
    from torch.cuda.amp import GradScaler, autocast

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    USE_AMP = True
else:
    USE_AMP = False
    print("CUDA not available, using CPU mode")

warnings.filterwarnings('ignore')


# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


class BrainTumorDataset(Dataset):
    """Optimized Dataset for both CPU and GPU"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = []
        self.labels = []

        # Load all images and labels
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for img_path in list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')):
                    self.images.append(str(img_path))
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ImprovedModel(nn.Module):
    """High-accuracy model with progressive improvements"""

    def __init__(self, num_classes=4, dropout_rate=0.3):
        super(ImprovedModel, self).__init__()

        # Use EfficientNet-B0 as backbone
        self.backbone = models.efficientnet_b0(pretrained=True)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        # SE block for attention
        self.se_block = SEBlock(1280)

        # Store initial dropout rate
        self.initial_dropout = dropout_rate

        # Multi-layer classifier with progressive dropout
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.4),

            nn.Linear(128, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.backbone.features(x)
        features = self.se_block(features)
        features = F.adaptive_avg_pool2d(features, 1)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output


class AdvancedEnsemble(nn.Module):
    """Advanced ensemble for pushing towards 99%+ accuracy"""

    def __init__(self, num_classes=4):
        super(AdvancedEnsemble, self).__init__()

        # Three diverse models with different dropout rates
        self.model1 = ImprovedModel(num_classes, dropout_rate=0.25)
        self.model2 = ImprovedModel(num_classes, dropout_rate=0.35)
        self.model3 = ImprovedModel(num_classes, dropout_rate=0.45)

        # Learnable ensemble weights
        self.weights = nn.Parameter(torch.ones(3) / 3)

        # Meta-learner for final prediction
        self.meta_learner = nn.Sequential(
            nn.Linear(num_classes * 3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        out3 = self.model3(x)

        # Weighted ensemble
        weights = F.softmax(self.weights, dim=0)
        ensemble_out = weights[0] * out1 + weights[1] * out2 + weights[2] * out3

        # Meta-learner combination
        concat_out = torch.cat([out1, out2, out3], dim=1)
        meta_out = self.meta_learner(concat_out)

        # Final combination
        final_out = 0.7 * ensemble_out + 0.3 * meta_out

        return final_out


class FocalLoss(nn.Module):
    """Focal Loss for handling hard examples"""

    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class PersistentTrainer:
    """Trainer that pushes through plateaus to reach maximum accuracy"""

    def __init__(self, num_classes=4, learning_rate=3e-4, use_ensemble=False):
        # Device selection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        print(f"🖥️ Using device: {self.device} ({device_name})")

        self.num_classes = num_classes
        self.use_ensemble = use_ensemble
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

        # Model selection
        if use_ensemble:
            self.model = AdvancedEnsemble(num_classes)
            print("📦 Using Advanced Ensemble Model")
        else:
            self.model = ImprovedModel(num_classes)
            print("📦 Using Improved Single Model")

        self.model = self.model.to(self.device)

        # Loss functions
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.focal_loss = FocalLoss(alpha=1, gamma=2)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
            amsgrad=True
        )

        # Cosine Annealing with Warm Restarts for breaking plateaus
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Initial restart period
            T_mult=2,  # Period doubling after each restart
            eta_min=1e-7
        )

        # Mixed precision scaler for GPU
        if USE_AMP:
            self.scaler = GradScaler()
        else:
            self.scaler = None

        # Training history
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rates': []
        }

        # Tracking variables
        self.best_val_acc = 0
        self.best_val_loss = float('inf')
        self.plateau_count = 0
        self.max_plateau_count = 50  # Will try 50 epochs even if stuck
        self.stuck_at_99 = False
        self.epochs_at_99 = 0
        self.restart_count = 0

    def mixup_data(self, x, y, alpha=0.3):
        """MixUp augmentation"""
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def cutmix_data(self, x, y, alpha=1.0):
        """CutMix augmentation"""
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)

        W = x.size()[2]
        H = x.size()[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        y_a, y_b = y, y[index]

        return x, y_a, y_b, lam

    def train_epoch(self, train_loader, epoch):
        """Training epoch with adaptive strategies"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Adaptive augmentation based on whether we're stuck
        if self.stuck_at_99:
            # When stuck at 99%, reduce augmentation to allow fine-tuning
            use_mixup_prob = 0.1
            use_cutmix_prob = 0.05
        elif self.best_val_acc >= 98:
            use_mixup_prob = 0.2
            use_cutmix_prob = 0.1
        else:
            use_mixup_prob = 0.3
            use_cutmix_prob = 0.2

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}', ncols=100)

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Randomly choose augmentation
            r = np.random.random()
            if r < use_mixup_prob:
                images, labels_a, labels_b, lam = self.mixup_data(images, labels)
                mixed_type = 'mixup'
            elif r < (use_mixup_prob + use_cutmix_prob):
                images, labels_a, labels_b, lam = self.cutmix_data(images, labels)
                mixed_type = 'cutmix'
            else:
                mixed_type = None

            if USE_AMP and self.scaler:  # GPU with mixed precision
                with autocast():
                    outputs = self.model(images)
                    if mixed_type:
                        ce_loss = lam * self.criterion(outputs, labels_a) + \
                                  (1 - lam) * self.criterion(outputs, labels_b)
                        focal = lam * self.focal_loss(outputs, labels_a) + \
                                (1 - lam) * self.focal_loss(outputs, labels_b)
                    else:
                        ce_loss = self.criterion(outputs, labels)
                        focal = self.focal_loss(outputs, labels)

                    # Adaptive loss weighting
                    if self.stuck_at_99:
                        # Focus more on CE loss when fine-tuning
                        loss = 0.95 * ce_loss + 0.05 * focal
                    elif self.best_val_acc >= 98:
                        loss = 0.8 * ce_loss + 0.2 * focal
                    else:
                        loss = 0.7 * ce_loss + 0.3 * focal
            else:  # CPU or GPU without AMP
                outputs = self.model(images)
                if mixed_type:
                    ce_loss = lam * self.criterion(outputs, labels_a) + \
                              (1 - lam) * self.criterion(outputs, labels_b)
                    focal = lam * self.focal_loss(outputs, labels_a) + \
                            (1 - lam) * self.focal_loss(outputs, labels_b)
                else:
                    ce_loss = self.criterion(outputs, labels)
                    focal = self.focal_loss(outputs, labels)

                if self.stuck_at_99:
                    loss = 0.95 * ce_loss + 0.05 * focal
                elif self.best_val_acc >= 98:
                    loss = 0.8 * ce_loss + 0.2 * focal
                else:
                    loss = 0.7 * ce_loss + 0.3 * focal

            if not mixed_type:
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)

            if USE_AMP and self.scaler:  # GPU with AMP
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)

                # Adaptive gradient clipping
                max_grad_norm = 0.5 if self.stuck_at_99 else 1.0
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:  # CPU or GPU without AMP
                loss.backward()

                max_grad_norm = 0.5 if self.stuck_at_99 else 1.0
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)

                self.optimizer.step()

            running_loss += loss.item()
            current_acc = 100. * correct / total if total > 0 else 0

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.2f}%',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total if total > 0 else 0

        return epoch_loss, epoch_acc

    def validate(self, val_loader):
        """Validation with detailed metrics"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        class_correct = list(0. for _ in range(self.num_classes))
        class_total = list(0. for _ in range(self.num_classes))

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validation', ncols=100):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if USE_AMP and self.scaler:  # GPU with AMP
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:  # CPU or GPU without AMP
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Per-class accuracy
                c = (predicted == labels).squeeze()
                for i in range(labels.size(0)):
                    label = labels[i]
                    class_correct[label] += c[i].item() if c.dim() == 0 else c[i].item()
                    class_total[label] += 1

        val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total

        # Print per-class accuracy if overall accuracy is high
        if val_acc >= 97:
            print("\n📊 Per-class accuracy:")
            for i in range(self.num_classes):
                if class_total[i] > 0:
                    acc = 100 * class_correct[i] / class_total[i]
                    print(f"  {self.classes[i]}: {acc:.2f}%")

        return val_loss, val_acc

    def train(self, train_loader, val_loader, epochs=200):
        """Persistent training that pushes through 99.07% plateau"""

        print(f"\n{'=' * 70}")
        print(f"🚀 PERSISTENT TRAINING - BREAKING THROUGH PLATEAUS")
        print(f"{'=' * 70}")
        print(f"Model: {'Advanced Ensemble' if self.use_ensemble else 'Improved Single'}")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {'Enabled' if USE_AMP else 'Disabled'}")
        print(f"Strategy: Will continue training even when stuck")
        print(f"{'=' * 70}\n")

        for epoch in range(1, epochs + 1):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, epoch)

            # Validation
            val_loss, val_acc = self.validate(val_loader)

            # Learning rate scheduling
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)

            # Print results
            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch}/{epochs}")
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.2e}")

            # Check if stuck at 99.07%
            if 99.0 <= val_acc <= 99.1:
                self.epochs_at_99 += 1
                if self.epochs_at_99 > 5:
                    self.stuck_at_99 = True
                    print(f"⚠️ Stuck at ~99.07% for {self.epochs_at_99} epochs")

                    # Try to break through the plateau
                    if self.epochs_at_99 % 10 == 0:
                        print("🔄 Attempting to break plateau with learning rate restart")
                        self.restart_count += 1
                        # Restart learning rate
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = 3e-4 * (0.5 ** self.restart_count)
                        print(f"   New learning rate: {param_group['lr']:.2e}")

                        # Reduce dropout slightly
                        if hasattr(self.model, 'classifier'):
                            for module in self.model.modules():
                                if isinstance(module, nn.Dropout):
                                    module.p = max(module.p * 0.9, 0.05)

            else:
                self.epochs_at_99 = 0
                if val_acc > 99.1:
                    self.stuck_at_99 = False

            # Save best model
            if val_acc > self.best_val_acc:
                improvement = val_acc - self.best_val_acc
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.plateau_count = 0
                self.save_model('best_model.pth')
                print(f"✅ New best model! Accuracy: {val_acc:.2f}% (+{improvement:.2f}%)")

                # Special notifications
                if val_acc >= 99.5:
                    print(f"\n🏆 BREAKTHROUGH! Achieved {val_acc:.2f}% accuracy!")
                    self.save_model(f'model_breakthrough_{val_acc:.2f}.pth')
                elif val_acc >= 99.1:
                    print(f"\n🎉 Pushed past 99.07%! New accuracy: {val_acc:.2f}%")
                    self.save_model(f'model_99plus_{val_acc:.2f}.pth')
            else:
                self.plateau_count += 1
                if self.plateau_count % 10 == 0:
                    print(f"📊 Plateau for {self.plateau_count} epochs")

            # Continue training regardless of plateau
            print(f"📈 Will continue training (Plateau count: {self.plateau_count}/{self.max_plateau_count})")

            # Only stop if accuracy is very low or we've tried for very long
            if epoch > 30 and val_acc < 70:
                print("\n⚠️ Training seems stuck at low accuracy")
                break
            elif self.plateau_count >= self.max_plateau_count and val_acc < 98:
                print("\n⏹️ Stopping after maximum plateau count")
                break

            print(f"{'=' * 70}")

            # Clear cache periodically
            if torch.cuda.is_available() and epoch % 10 == 0:
                torch.cuda.empty_cache()
            gc.collect()

        # Final summary
        print(f"\n{'=' * 70}")
        print(f"📈 TRAINING SUMMARY")
        print(f"{'=' * 70}")
        print(f"Total Epochs Trained: {len(self.history['train_acc'])}")
        print(f"Best Validation Accuracy: {self.best_val_acc:.2f}%")
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")
        print(f"Epochs stuck at ~99%: {self.epochs_at_99}")
        print(f"Learning rate restarts: {self.restart_count}")

        if self.best_val_acc >= 99.5:
            print(f"🌟 EXCEPTIONAL: Broke through to {self.best_val_acc:.2f}%!")
        elif self.best_val_acc > 99.07:
            print(f"✅ SUCCESS: Pushed past 99.07% to {self.best_val_acc:.2f}%!")
        else:
            print(f"📊 Final accuracy: {self.best_val_acc:.2f}%")

        # Save training history
        self.save_history()

        return self.best_val_acc

    def save_model(self, filename):
        """Save model checkpoint"""
        save_dir = Path('experiments/saved_models')
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'epochs_at_99': self.epochs_at_99,
            'plateau_count': self.plateau_count
        }

        save_path = save_dir / filename
        torch.save(checkpoint, save_path)
        print(f"💾 Model saved: {save_path}")

    def save_history(self):
        """Save training history to JSON"""
        history_path = Path('experiments/training_history.json')
        history_path.parent.mkdir(exist_ok=True)

        history_data = {
            'best_val_acc': float(self.best_val_acc),
            'best_val_loss': float(self.best_val_loss),
            'epochs_trained': len(self.history['train_acc']),
            'epochs_at_99': self.epochs_at_99,
            'plateau_count': self.plateau_count,
            'history': {
                'train_loss': [float(x) for x in self.history['train_loss']],
                'train_acc': [float(x) for x in self.history['train_acc']],
                'val_loss': [float(x) for x in self.history['val_loss']],
                'val_acc': [float(x) for x in self.history['val_acc']],
                'learning_rates': [float(x) for x in self.history['learning_rates']]
            }
        }

        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        print(f"📊 Training history saved: {history_path}")


def create_advanced_transforms():
    """Advanced transforms for high accuracy"""

    train_transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Persistent Training to Break Through 99% Plateau')
    parser.add_argument('--epochs', type=int, default=200, help='Maximum epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--use-ensemble', action='store_true', help='Use ensemble model')
    parser.add_argument('--data-dir', type=str, default='dataset_processed')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🧠 BRAIN TUMOR CLASSIFICATION - PLATEAU BREAKER")
    print("=" * 70)
    print(f"Device: {'GPU (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    print(f"PyTorch: {PYTORCH_VERSION}")
    print(f"Target: Break through 99.07% plateau")
    print("=" * 70)

    # Create transforms
    train_transform, val_transform = create_advanced_transforms()

    # Load datasets
    train_data_dir = Path(args.data_dir) / 'Training'
    test_data_dir = Path(args.data_dir) / 'Testing'

    if not train_data_dir.exists():
        print(f"❌ Error: Data not found at {train_data_dir}")
        print("Please run: python preprocess_dataset.py first")
        return

    print("\n📁 Loading datasets...")
    full_train_dataset = BrainTumorDataset(train_data_dir, transform=train_transform)
    test_dataset = BrainTumorDataset(test_data_dir, transform=val_transform)

    # Split dataset (85-15 for more training data)
    train_size = int(0.85 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=generator
    )

    # Adjust batch size for CPU if needed
    if not torch.cuda.is_available():
        args.batch_size = min(args.batch_size, 16)
        args.num_workers = min(args.num_workers, 2)
        print(f"📝 Adjusted for CPU: batch_size={args.batch_size}, workers={args.num_workers}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if args.num_workers > 0 else False,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if args.num_workers > 0 else False
    )

    print(f"\n📊 Dataset Statistics:")
    print(f"  Training: {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"  Validation: {len(val_dataset)} samples ({len(val_loader)} batches)")
    print(f"  Testing: {len(test_dataset)} samples")
    print(f"  Batch Size: {args.batch_size}")

    # Initialize trainer
    trainer = PersistentTrainer(
        num_classes=4,
        learning_rate=args.lr,
        use_ensemble=args.use_ensemble
    )

    # Start training
    try:
        best_acc = trainer.train(train_loader, val_loader, epochs=args.epochs)

        print(f"\n{'=' * 70}")
        print(f"🏆 FINAL RESULTS")
        print(f"{'=' * 70}")
        print(f"Best Validation Accuracy: {best_acc:.2f}%")

        if best_acc >= 99.5:
            print("🌟 EXCEPTIONAL! Broke through to 99.5%+ accuracy!")
        elif best_acc > 99.07:
            print("✅ SUCCESS! Pushed past the 99.07% plateau!")
        elif best_acc >= 99.0:
            print("📈 Good result at 99%+, consider using --use-ensemble for breakthrough")
        else:
            print(f"📊 Achieved {best_acc:.2f}% accuracy")

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        trainer.save_model('interrupted_checkpoint.pth')
        trainer.save_history()
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        trainer.save_model('error_checkpoint.pth')
        trainer.save_history()
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("\n🧹 Cleanup complete")


if __name__ == '__main__':
    main()