import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import wandb


class ClassifierTrainer:
    def __init__(self, model, config, logger):
        self.model = model
        self.config = config
        self.logger = logger
        self.device = torch.device(config.DEVICE)

        self.model = self.model.to(self.device)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE_CLF,
            weight_decay=1e-4
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=5,
            factor=0.5
        )

        # Best model tracking
        self.best_val_acc = 0
        self.best_model_state = None

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        with tqdm(train_loader, desc='Training') as pbar:
            for batch_idx, data in enumerate(pbar):
                if len(data) == 3:
                    images, labels, _ = data
                else:
                    images, labels = data

                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs, _ = self.model(images)
                loss = self.criterion(outputs, labels)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                # Track metrics
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{accuracy_score(all_labels, all_preds):.4f}'
                })

        # Calculate epoch metrics
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')

        return epoch_loss, epoch_acc, epoch_f1

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            with tqdm(val_loader, desc='Validation') as pbar:
                for data in pbar:
                    if len(data) == 3:
                        images, labels, _ = data
                    else:
                        images, labels = data

                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    outputs, _ = self.model(images)
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

        # Calculate AUROC
        all_probs = np.array(all_probs)
        val_auroc = roc_auc_score(
            all_labels,
            all_probs,
            multi_class='ovr',
            average='weighted'
        )

        return val_loss, val_acc, val_f1, val_auroc

    def train(self, train_loader, val_loader, epochs):
        self.logger.info("Starting training...")

        for epoch in range(epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{epochs}")

            # Training
            train_loss, train_acc, train_f1 = self.train_epoch(train_loader)

            # Validation
            val_loss, val_acc, val_f1, val_auroc = self.validate(val_loader)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Logging
            self.logger.info(
                f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}"
            )
            self.logger.info(
                f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
                f"F1: {val_f1:.4f}, AUROC: {val_auroc:.4f}"
            )

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                torch.save(
                    self.best_model_state,
                    self.config.MODEL_DIR / 'best_classifier.pth'
                )
                self.logger.info(f"Saved best model with val_acc: {val_acc:.4f}")

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

        return self.best_model_state