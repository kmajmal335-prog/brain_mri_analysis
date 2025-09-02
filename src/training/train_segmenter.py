import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from ..training.loss_functions import CombinedLoss


class SegmenterTrainer:
    def __init__(self, model, config, logger):
        self.model = model
        self.config = config
        self.logger = logger
        self.device = torch.device(config.DEVICE)

        self.model = self.model.to(self.device)

        # Loss and optimizer
        self.criterion = CombinedLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE_SEG
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.EPOCHS_SEG
        )

        self.best_dice = 0
        self.best_model_state = None

    def calculate_dice(self, pred, target):
        smooth = 1e-6
        pred = (pred > 0.5).float()
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return dice.item()

    def calculate_sensitivity(self, pred, target):
        pred = (pred > 0.5).float()
        true_positives = (pred * target).sum()
        actual_positives = target.sum()
        sensitivity = true_positives / (actual_positives + 1e-6)
        return sensitivity.item()

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        total_dice = 0
        total_sensitivity = 0
        num_batches = 0

        with tqdm(train_loader, desc='Training Segmentation') as pbar:
            for batch_idx, data in enumerate(pbar):
                if len(data) == 3:
                    images, _, masks = data
                else:
                    continue  # Skip if no masks

                if masks is None:
                    continue

                images = images.to(self.device)
                masks = masks.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss, loss_dict = self.criterion(outputs, masks)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Calculate metrics
                dice = self.calculate_dice(outputs, masks)
                sensitivity = self.calculate_sensitivity(outputs, masks)

                total_loss += loss.item()
                total_dice += dice
                total_sensitivity += sensitivity
                num_batches += 1

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'dice': f'{dice:.4f}',
                    'sens': f'{sensitivity:.4f}'
                })

        if num_batches == 0:
            return 0, 0, 0

        epoch_loss = total_loss / num_batches
        epoch_dice = total_dice / num_batches
        epoch_sensitivity = total_sensitivity / num_batches

        return epoch_loss, epoch_dice, epoch_sensitivity

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        total_dice = 0
        total_sensitivity = 0
        num_batches = 0

        with torch.no_grad():
            with tqdm(val_loader, desc='Validation Segmentation') as pbar:
                for data in pbar:
                    if len(data) == 3:
                        images, _, masks = data
                    else:
                        continue

                    if masks is None:
                        continue

                    images = images.to(self.device)
                    masks = masks.to(self.device)

                    outputs = self.model(images)
                    loss, _ = self.criterion(outputs, masks)

                    dice = self.calculate_dice(outputs, masks)
                    sensitivity = self.calculate_sensitivity(outputs, masks)

                    total_loss += loss.item()
                    total_dice += dice
                    total_sensitivity += sensitivity
                    num_batches += 1

        if num_batches == 0:
            return 0, 0, 0

        val_loss = total_loss / num_batches
        val_dice = total_dice / num_batches
        val_sensitivity = total_sensitivity / num_batches

        return val_loss, val_dice, val_sensitivity

    def train(self, train_loader, val_loader, epochs):
        self.logger.info("Starting segmentation training...")

        for epoch in range(epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{epochs}")

            # Training
            train_loss, train_dice, train_sens = self.train_epoch(train_loader)

            # Validation
            val_loss, val_dice, val_sens = self.validate(val_loader)

            # Learning rate scheduling
            self.scheduler.step()

            # Logging
            self.logger.info(
                f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, "
                f"Sensitivity: {train_sens:.4f}"
            )
            self.logger.info(
                f"Val - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, "
                f"Sensitivity: {val_sens:.4f}"
            )

            # Save best model
            if val_dice > self.best_dice:
                self.best_dice = val_dice
                self.best_model_state = self.model.state_dict().copy()
                torch.save(
                    self.best_model_state,
                    self.config.MODEL_DIR / 'best_segmenter.pth'
                )
                self.logger.info(f"Saved best model with val_dice: {val_dice:.4f}")

        return self.best_model_state