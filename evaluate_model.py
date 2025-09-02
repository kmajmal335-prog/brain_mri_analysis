import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import json
from pathlib import Path
from PIL import Image
import warnings

warnings.filterwarnings('ignore')


# Import the same model architecture used in training
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
    """Same model architecture as used in training"""

    def __init__(self, num_classes=4, dropout_rate=0.3):
        super(ImprovedModel, self).__init__()

        # Use EfficientNet-B0 as backbone
        self.backbone = models.efficientnet_b0(pretrained=False)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        # SE block for attention
        self.se_block = SEBlock(1280)

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

    def forward(self, x):
        features = self.backbone.features(x)
        features = self.se_block(features)
        features = torch.nn.functional.adaptive_avg_pool2d(features, 1)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output


class BrainTumorDataset(Dataset):
    """Dataset for Brain Tumor Classification"""

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


class ModelEvaluator:
    """Comprehensive model evaluation"""

    def __init__(self, model_path='experiments/saved_models/best_model.pth'):
        self.model_path = Path(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.num_classes = len(self.classes)

        # Load model
        self.model = self._load_model()

        # Results storage
        self.results = {}

    def _load_model(self):
        """Load the trained model with correct architecture"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Check if it's an ensemble model
        use_ensemble = checkpoint.get('use_ensemble', False)

        if use_ensemble:
            # Handle ensemble model (simplified for now)
            print("⚠️ Ensemble model detected. Loading single model for evaluation.")
            model = ImprovedModel(num_classes=self.num_classes)
        else:
            # Create the same model architecture used in training
            model = ImprovedModel(num_classes=self.num_classes)

        # Load weights
        if 'model_state_dict' in checkpoint:
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except RuntimeError as e:
                print(f"⚠️ Model architecture mismatch. Attempting partial load...")
                # Try to load what we can
                model_dict = model.state_dict()
                pretrained_dict = checkpoint['model_state_dict']

                # Filter out keys that don't match
                pretrained_dict = {k: v for k, v in pretrained_dict.items()
                                   if k in model_dict and model_dict[k].shape == v.shape}

                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict, strict=False)
                print(f"✅ Loaded {len(pretrained_dict)}/{len(model_dict)} parameters")
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        model.eval()

        # Print training info if available
        if 'best_val_acc' in checkpoint:
            print(f"📊 Model trained to: {checkpoint['best_val_acc']:.2f}% validation accuracy")
        if 'epochs_trained' in checkpoint:
            print(f"📊 Epochs trained: {checkpoint.get('epochs_trained', 'Unknown')}")

        print(f"✅ Model loaded from {self.model_path}")
        return model

    def prepare_test_data(self, data_dir='dataset_processed/Testing'):
        """Prepare test data loader"""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        test_dataset = BrainTumorDataset(data_dir, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        print(f"📊 Test data loaded: {len(test_dataset)} images")
        return test_loader, test_dataset

    def evaluate(self, test_loader):
        """Perform comprehensive evaluation"""
        print("\n🔬 Evaluating model...")

        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)

                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)

        # Calculate metrics
        self.results['predictions'] = all_predictions
        self.results['labels'] = all_labels
        self.results['probabilities'] = all_probabilities

        # Accuracy
        accuracy = np.mean(all_predictions == all_labels)
        self.results['accuracy'] = accuracy

        # Classification report
        report = classification_report(
            all_labels, all_predictions,
            target_names=self.classes,
            output_dict=True
        )
        self.results['classification_report'] = report

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        self.results['confusion_matrix'] = cm

        # ROC-AUC (multi-class)
        labels_binarized = label_binarize(all_labels, classes=list(range(self.num_classes)))
        try:
            auc_scores = {}
            for i, class_name in enumerate(self.classes):
                auc = roc_auc_score(labels_binarized[:, i], all_probabilities[:, i])
                auc_scores[class_name] = auc

            # Average AUC
            avg_auc = np.mean(list(auc_scores.values()))
            self.results['auc_scores'] = auc_scores
            self.results['avg_auc'] = avg_auc
        except:
            self.results['auc_scores'] = {}
            self.results['avg_auc'] = 0

        return self.results

    def plot_confusion_matrix(self, save_path='results/confusion_matrix.png'):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))

        cm = self.results['confusion_matrix']

        # Create heatmap manually without seaborn
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()

        # Add text annotations
        for i in range(len(self.classes)):
            for j in range(len(self.classes)):
                plt.text(j, i, str(cm[i, j]),
                         ha='center', va='center',
                         color='white' if cm[i, j] > cm.max() / 2 else 'black')

        plt.title('Confusion Matrix', fontsize=16)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(range(len(self.classes)), self.classes, rotation=45)
        plt.yticks(range(len(self.classes)), self.classes)

        # Save
        Path(save_path).parent.mkdir(exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

        print(f"📊 Confusion matrix saved to {save_path}")

    def print_results(self):
        """Print evaluation results"""
        print("\n" + "=" * 60)
        print("📊 EVALUATION RESULTS")
        print("=" * 60)

        print(f"\n✅ Overall Accuracy: {self.results['accuracy']:.4f} ({self.results['accuracy'] * 100:.2f}%)")

        if 'avg_auc' in self.results:
            print(f"✅ Average ROC-AUC: {self.results['avg_auc']:.4f}")

        print("\n📈 Per-Class Metrics:")
        print("-" * 40)

        report = self.results['classification_report']
        for class_name in self.classes:
            if class_name in report:
                metrics = report[class_name]
                print(f"\n{class_name.upper()}:")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  F1-Score: {metrics['f1-score']:.4f}")
                print(f"  Support: {metrics['support']}")

                if class_name in self.results.get('auc_scores', {}):
                    print(f"  ROC-AUC: {self.results['auc_scores'][class_name]:.4f}")

        print("\n📊 Weighted Averages:")
        print("-" * 40)
        if 'weighted avg' in report:
            avg_metrics = report['weighted avg']
            print(f"  Precision: {avg_metrics['precision']:.4f}")
            print(f"  Recall: {avg_metrics['recall']:.4f}")
            print(f"  F1-Score: {avg_metrics['f1-score']:.4f}")

    def save_results(self, save_path='results/evaluation_results.json'):
        """Save evaluation results to JSON"""
        Path(save_path).parent.mkdir(exist_ok=True)

        # Prepare serializable results
        save_dict = {
            'accuracy': float(self.results['accuracy']),
            'avg_auc': float(self.results.get('avg_auc', 0)),
            'classification_report': self.results['classification_report'],
            'confusion_matrix': self.results['confusion_matrix'].tolist(),
            'auc_scores': self.results.get('auc_scores', {}),
            'model_path': str(self.model_path),
            'num_test_samples': len(self.results['labels'])
        }

        with open(save_path, 'w') as f:
            json.dump(save_dict, f, indent=2)

        print(f"\n📄 Results saved to {save_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate Brain Tumor Classification Model')
    parser.add_argument('--model', type=str, default='experiments/saved_models/best_model.pth',
                        help='Path to trained model')
    parser.add_argument('--data-dir', type=str, default='dataset_processed/Testing',
                        help='Test data directory')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save visualization plots')

    args = parser.parse_args()

    # Check if model exists
    if not Path(args.model).exists():
        # Try alternative paths
        alt_paths = [
            'experiments/saved_models/best_model.pth',
            'experiments/saved_models/best_99plus_model.pth',
            'experiments/saved_models/model_98.5percent.pth',
            'experiments/saved_models/final_enhanced_model.pth',
            'experiments/saved_models/best_cpu_model.pth'
        ]

        model_found = False
        for alt_path in alt_paths:
            if Path(alt_path).exists():
                args.model = alt_path
                model_found = True
                print(f"📌 Using model: {alt_path}")
                break

        if not model_found:
            print("❌ Error: No trained model found!")
            print("Available models in experiments/saved_models:")
            model_dir = Path('experiments/saved_models')
            if model_dir.exists():
                for model_file in model_dir.glob('*.pth'):
                    print(f"  - {model_file}")
            print("\nPlease run: python train_model.py first")
            return

    # Check if test data exists
    if not Path(args.data_dir).exists():
        print(f"❌ Error: Test data not found at '{args.data_dir}'")
        print("Please run: python preprocess_dataset.py first")
        return

    # Initialize evaluator
    evaluator = ModelEvaluator(model_path=args.model)

    # Prepare test data
    try:
        test_loader, test_dataset = evaluator.prepare_test_data(args.data_dir)
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return

    # Evaluate model
    results = evaluator.evaluate(test_loader)

    # Print results
    evaluator.print_results()

    # Save results
    evaluator.save_results()

    # Generate plots
    if args.save_plots:
        evaluator.plot_confusion_matrix()

    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()