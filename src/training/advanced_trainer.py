import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm
import wandb


class AdvancedTrainer:
    """Advanced training with mixed precision and advanced techniques"""

    def __init__(self, model, config, logger):
        self.model = model
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move model to device
        self.model = self.model.to(self.device)

        # Loss functions
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.auxiliary_criterion = nn.CrossEntropyLoss()

        # Optimizer with different learning rates for different parts
        self.optimizer = self._create_optimizer()

        # Learning rate schedulers
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

        # Mixed precision training
        self.scaler = GradScaler()

        # Best model tracking
        self.best_val_acc = 0
        self.best_val_f1 = 0
        self.patience_counter = 0
        self.early_stopping_patience = 15

    def _create_optimizer(self):
        """Create optimizer with different learning rates"""
        # Different learning rates for different parts
        params = [
            {'params': self.model.model1.backbone.parameters(), 'lr': 1e-5},
            {'params': self.model.model1.classifier.parameters(), 'lr': 1e-3},
            {'params': self.model.model2.parameters(), 'lr': 5e-5},
            {'params': self.model.model3.parameters(), 'lr': 5e-5},
            {'params': self.model.model4.parameters(), 'lr': 5e-5},
            {'params': self.model.model5.parameters(), 'lr': 5e-5},
            {'params': self.model.meta_learner.parameters(), 'lr': 1e-3},
            {'params': [self.model.ensemble_weights], 'lr': 1e-2}
        ]

        return optim.AdamW(params, weight_decay=1e-4)

    def mixup_data(self, x, y, alpha=1.0):
        """Mixup augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
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
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

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
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(train_loader, desc=f'Training Epoch {epoch}')

        for batch_idx, data in enumerate(pbar):
            if len(data) == 3:
                images, labels, _ = data
            else:
                images, labels = data

            images = images.to(self.device)
            labels = labels.to(self.device)

            # Apply MixUp or CutMix randomly
            r = np.random.rand(1)
            if r < 0.3:
                images, labels_a, labels_b, lam = self.mixup_data(images, labels, alpha=0.4)
                mixed = True
            elif r < 0.6:
                images, labels_a, labels_b, lam = self.cutmix_data(images, labels, alpha=1.0)
                mixed = True
            else:
                mixed = False

            # Mixed precision training
            with autocast():
                outputs, ensemble_prob, meta_output, attention = self.model(images)

                if mixed:
                    loss = lam * self.criterion(outputs, labels_a) + (1 - lam) * self.criterion(outputs, labels_b)
                else:
                    loss = self.criterion(outputs, labels)

                # Add auxiliary loss if available
                if hasattr(self.model.model1, 'aux_classifier'):
                    if mixed:
                        aux_loss = lam * self.auxiliary_criterion(meta_output, labels_a) + \
                                   (1 - lam) * self.auxiliary_criterion(meta_output, labels_b)
                    else:
                        aux_loss = self.auxiliary_criterion(meta_output, labels)
                    loss = 0.7 * loss + 0.3 * aux_loss

            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Track metrics
            total_loss += loss.item()

            if not mixed:
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })

        # Calculate epoch metrics
        epoch_loss = total_loss / len(train_loader)

        if len(all_preds) > 0:
            epoch_acc = accuracy_score(all_labels, all_preds)
            epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
        else:
            epoch_acc = 0
            epoch_f1 = 0

        return epoch_loss, epoch_acc, epoch_f1

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')

            for data in pbar:
                if len(data) == 3:
                    images, labels, _ = data
                else:
                    images, labels = data

                images = images.to(self.device)
                labels = labels.to(self.device)

                with autocast():
                    outputs, _, _, _ = self.model(images)
                    loss = self.criterion(outputs, labels)

                total_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Calculate metrics
        val_loss = total_loss / len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')

        # Calculate per-class accuracy
        cm = confusion_matrix(all_labels, all_preds)
        per_class_acc = cm.diagonal() / cm.sum(axis=1)

        # ROC-AUC
        all_probs = np.array(all_probs)
        try:
            val_auroc = roc_auc_score(
                all_labels,
                all_probs,
                multi_class='ovr',
                average='weighted'
            )
        except:
            val_auroc = 0

        return val_loss, val_acc, val_f1, val_auroc, per_class_acc

    def train(self, train_loader, val_loader, epochs):
        self.logger.info("Starting advanced training with ensemble model...")

        for epoch in range(epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{epochs}")

            # Training
            train_loss, train_acc, train_f1 = self.train_epoch(train_loader, epoch)

            # Validation
            val_loss, val_acc, val_f1, val_auroc, per_class_acc = self.validate(val_loader)

            # Learning rate scheduling
            self.scheduler.step()

            # Logging
            self.logger.info(
                f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}"
            )
            self.logger.info(
                f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
                f"F1: {val_f1:.4f}, AUROC: {val_auroc:.4f}"
            )
            self.logger.info(f"Per-class accuracy: {per_class_acc}")

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_f1 = val_f1
                self.patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'val_auroc': val_auroc
                }, self.config.MODEL_DIR / 'best_ensemble_model.pth')

                self.logger.info(f"✅ Saved best model with val_acc: {val_acc:.4f}")
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                self.logger.info("Early stopping triggered!")
                break

            # Log to wandb
            if wandb.run:
                wandb.log({
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'train_f1': train_f1,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'val_auroc': val_auroc,
                    'lr': self.optimizer.param_groups[0]['lr']
                })

        self.logger.info(f"\nBest validation accuracy: {self.best_val_acc:.4f}")
        self.logger.info(f"Best validation F1: {self.best_val_f1:.4f}")

        return self.model