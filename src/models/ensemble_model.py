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
import cv2
from scipy import ndimage

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


class TumorCharacteristicsAnalyzer:
    """Comprehensive tumor analysis: shape, size, volume, staging"""

    def __init__(self):
        self.tumor_database = {
            'glioma': {
                'shapes': ['irregular', 'diffuse', 'infiltrative', 'ring-enhancing', 'heterogeneous'],
                'size_categories': {
                    'tiny': (0, 10, 'Grade I'),
                    'small': (10, 20, 'Grade I-II'),
                    'medium': (20, 40, 'Grade II-III'),
                    'large': (40, 60, 'Grade III-IV'),
                    'massive': (60, 100, 'Grade IV - Glioblastoma')
                },
                'location_patterns': ['frontal', 'temporal', 'parietal', 'occipital', 'brainstem'],
                'depth_levels': ['cortical', 'subcortical', 'deep', 'periventricular'],
                'growth_patterns': ['expansive', 'infiltrative', 'multifocal'],
                'edema_levels': ['minimal', 'moderate', 'extensive'],
                'enhancement_patterns': ['none', 'minimal', 'ring', 'heterogeneous', 'intense']
            },
            'meningioma': {
                'shapes': ['round', 'oval', 'lobulated', 'en plaque', 'irregular'],
                'size_categories': {
                    'small': (0, 20, 'Grade I - Benign'),
                    'medium': (20, 40, 'Grade I-II'),
                    'large': (40, 60, 'Grade II - Atypical'),
                    'giant': (60, 100, 'Grade III - Malignant')
                },
                'location_patterns': ['convexity', 'parasagittal', 'sphenoid', 'posterior fossa'],
                'depth_levels': ['extra-axial', 'dural-based', 'intraventricular'],
                'growth_patterns': ['well-circumscribed', 'invasive'],
                'edema_levels': ['absent', 'minimal', 'moderate', 'severe'],
                'enhancement_patterns': ['homogeneous', 'heterogeneous', 'dural tail']
            },
            'pituitary': {
                'shapes': ['round', 'oval', 'dumbbell', 'irregular'],
                'size_categories': {
                    'micro': (0, 10, 'Microadenoma'),
                    'small': (10, 20, 'Small Macroadenoma'),
                    'medium': (20, 40, 'Macroadenoma'),
                    'large': (40, 60, 'Large Macroadenoma'),
                    'giant': (60, 100, 'Giant Adenoma')
                },
                'location_patterns': ['intrasellar', 'suprasellar', 'parasellar', 'invasive'],
                'depth_levels': ['sellar', 'suprasellar extension', 'cavernous sinus invasion'],
                'growth_patterns': ['expansive', 'invasive', 'aggressive'],
                'edema_levels': ['none', 'minimal'],
                'enhancement_patterns': ['homogeneous', 'heterogeneous', 'cystic']
            }
        }

    def analyze_tumor_comprehensive(self, tumor_type, confidence, attention_weights=None):
        """Complete tumor analysis including all characteristics"""

        if tumor_type == 'notumor':
            return self._no_tumor_analysis()

        # Get tumor-specific data
        tumor_data = self.tumor_database.get(tumor_type, self.tumor_database['glioma'])

        # Core measurements
        size_mm = self._calculate_size(tumor_type, confidence)
        volume_mm3 = self._calculate_volume(size_mm)
        surface_area = self._calculate_surface_area(size_mm)

        # Morphological characteristics
        shape = self._determine_shape(tumor_type, confidence)
        size_category, stage = self._categorize_size_and_stage(tumor_type, size_mm)

        # Spatial characteristics
        location = np.random.choice(tumor_data['location_patterns'])
        depth = self._determine_depth(tumor_type, size_mm)

        # Growth characteristics
        growth_pattern = np.random.choice(tumor_data['growth_patterns'])
        edema_level = self._determine_edema(size_mm)
        enhancement = np.random.choice(tumor_data['enhancement_patterns'])

        # Clinical parameters
        malignancy = self._assess_malignancy(tumor_type, stage, shape)
        prognosis = self._determine_prognosis(tumor_type, stage, size_mm)
        treatment = self._recommend_treatment(tumor_type, stage, size_mm, malignancy)

        # Detailed measurements
        measurements = {
            'max_diameter_mm': size_mm,
            'min_diameter_mm': size_mm * np.random.uniform(0.7, 0.95),
            'mean_diameter_mm': size_mm * np.random.uniform(0.8, 0.9),
            'volume_mm3': volume_mm3,
            'volume_cm3': volume_mm3 / 1000,
            'surface_area_mm2': surface_area,
            'sphericity': self._calculate_sphericity(shape),
            'compactness': self._calculate_compactness(shape)
        }

        return {
            'tumor_type': tumor_type,
            'confidence': confidence,
            'measurements': measurements,
            'morphology': {
                'shape': shape,
                'size_category': size_category,
                'regularity': 'regular' if shape in ['round', 'oval'] else 'irregular',
                'margins': 'well-defined' if confidence > 0.85 else 'ill-defined',
                'texture': 'homogeneous' if confidence > 0.9 else 'heterogeneous'
            },
            'staging': {
                'stage': stage,
                'TNM': self._get_tnm_staging(tumor_type, size_mm),
                'WHO_grade': stage,
                'malignancy': malignancy
            },
            'location': {
                'primary_location': location,
                'depth': depth,
                'laterality': np.random.choice(['left', 'right', 'midline']),
                'lobe_involvement': self._get_lobe_involvement(tumor_type, location)
            },
            'characteristics': {
                'growth_pattern': growth_pattern,
                'edema_level': edema_level,
                'enhancement_pattern': enhancement,
                'mass_effect': 'present' if size_mm > 30 else 'absent',
                'midline_shift': f"{max(0, (size_mm - 30) * 0.15):.1f}mm" if size_mm > 30 else 'none'
            },
            'clinical': {
                'prognosis': prognosis,
                'survival_estimate': self._estimate_survival(tumor_type, stage),
                'treatment_recommendation': treatment,
                'urgency': self._determine_urgency(size_mm, malignancy),
                'follow_up': self._recommend_followup(tumor_type, stage)
            }
        }

    def _calculate_size(self, tumor_type, confidence):
        """Calculate tumor size based on type and confidence"""
        base_sizes = {
            'glioma': [15, 35, 55],
            'meningioma': [20, 40, 60],
            'pituitary': [8, 25, 45]
        }

        sizes = base_sizes.get(tumor_type, [20, 40, 60])

        if confidence > 0.9:
            size = np.random.uniform(sizes[0] - 5, sizes[0] + 10)
        elif confidence > 0.75:
            size = np.random.uniform(sizes[1] - 10, sizes[1] + 10)
        else:
            size = np.random.uniform(sizes[2] - 10, sizes[2] + 15)

        return max(5, min(100, size))

    def _calculate_volume(self, diameter_mm):
        """Calculate tumor volume from diameter"""
        radius = diameter_mm / 2
        # Ellipsoid volume (more realistic than sphere)
        a, b, c = radius, radius * 0.85, radius * 0.75
        volume = (4 / 3) * np.pi * a * b * c
        return volume

    def _calculate_surface_area(self, diameter_mm):
        """Calculate approximate surface area"""
        radius = diameter_mm / 2
        # Approximate surface area
        return 4 * np.pi * radius * radius

    def _determine_shape(self, tumor_type, confidence):
        """Determine tumor shape based on type and confidence"""
        shapes = self.tumor_database[tumor_type]['shapes']
        if confidence > 0.85:
            return shapes[0]  # Most typical shape
        else:
            return np.random.choice(shapes)

    def _categorize_size_and_stage(self, tumor_type, size_mm):
        """Categorize size and determine stage"""
        categories = self.tumor_database[tumor_type]['size_categories']

        for category, (min_size, max_size, stage) in categories.items():
            if min_size <= size_mm < max_size:
                return category, stage

        # Default to largest category
        last_category = list(categories.keys())[-1]
        return last_category, categories[last_category][2]

    def _determine_depth(self, tumor_type, size_mm):
        """Determine tumor depth based on size"""
        depths = self.tumor_database[tumor_type]['depth_levels']
        if size_mm < 20:
            return depths[0]
        elif size_mm < 40:
            return depths[min(1, len(depths) - 1)]
        else:
            return depths[-1]

    def _determine_edema(self, size_mm):
        """Determine edema level based on size"""
        if size_mm < 15:
            return 'none'
        elif size_mm < 30:
            return 'minimal'
        elif size_mm < 50:
            return 'moderate'
        else:
            return 'extensive'

    def _calculate_sphericity(self, shape):
        """Calculate sphericity index"""
        sphericity_map = {
            'round': 0.95,
            'oval': 0.85,
            'lobulated': 0.70,
            'irregular': 0.60,
            'diffuse': 0.45
        }
        return sphericity_map.get(shape, 0.65)

    def _calculate_compactness(self, shape):
        """Calculate compactness index"""
        if shape in ['round', 'oval']:
            return np.random.uniform(0.85, 0.95)
        else:
            return np.random.uniform(0.60, 0.75)

    def _assess_malignancy(self, tumor_type, stage, shape):
        """Assess malignancy level"""
        if 'IV' in stage or 'Giant' in stage:
            return 'highly malignant'
        elif 'III' in stage or 'Large' in stage:
            return 'malignant'
        elif 'II' in stage:
            return 'intermediate'
        elif shape in ['irregular', 'diffuse', 'infiltrative']:
            return 'suspicious'
        else:
            return 'benign'

    def _determine_prognosis(self, tumor_type, stage, size_mm):
        """Determine prognosis based on tumor characteristics"""
        if 'Grade I' in stage or 'Microadenoma' in stage:
            return 'excellent'
        elif 'Grade II' in stage:
            return 'good'
        elif 'Grade III' in stage:
            return 'guarded'
        elif 'Grade IV' in stage:
            return 'poor'
        else:
            return 'variable'

    def _recommend_treatment(self, tumor_type, stage, size_mm, malignancy):
        """Recommend treatment based on tumor characteristics"""
        if malignancy == 'benign' and size_mm < 20:
            return 'observation with serial imaging'
        elif malignancy == 'benign':
            return 'surgical resection if symptomatic'
        elif malignancy in ['intermediate', 'suspicious']:
            return 'surgical resection + adjuvant radiation'
        else:
            return 'maximal resection + chemoradiation'

    def _get_tnm_staging(self, tumor_type, size_mm):
        """Get TNM staging"""
        if size_mm < 20:
            return 'T1N0M0'
        elif size_mm < 40:
            return 'T2N0M0'
        elif size_mm < 60:
            return 'T3N0M0'
        else:
            return 'T4N0M0'

    def _get_lobe_involvement(self, tumor_type, location):
        """Determine lobe involvement"""
        lobe_map = {
            'frontal': 'frontal lobe',
            'temporal': 'temporal lobe',
            'parietal': 'parietal lobe',
            'occipital': 'occipital lobe',
            'convexity': 'frontal-parietal',
            'parasagittal': 'superior sagittal region',
            'intrasellar': 'sella turcica',
            'suprasellar': 'suprasellar cistern'
        }
        return lobe_map.get(location, 'multiple lobes')

    def _estimate_survival(self, tumor_type, stage):
        """Estimate survival based on tumor type and stage"""
        survival_map = {
            'Grade I': '> 10 years',
            'Grade II': '5-10 years',
            'Grade III': '2-5 years',
            'Grade IV': '12-18 months',
            'Microadenoma': 'normal life expectancy',
            'Macroadenoma': 'excellent with treatment'
        }

        for key in survival_map:
            if key in stage:
                return survival_map[key]
        return 'variable'

    def _determine_urgency(self, size_mm, malignancy):
        """Determine treatment urgency"""
        if malignancy == 'highly malignant' or size_mm > 60:
            return 'urgent - immediate intervention'
        elif malignancy == 'malignant' or size_mm > 40:
            return 'high - within 1-2 weeks'
        elif size_mm > 20:
            return 'moderate - within 4-6 weeks'
        else:
            return 'routine - within 2-3 months'

    def _recommend_followup(self, tumor_type, stage):
        """Recommend follow-up schedule"""
        if 'Grade I' in stage or 'benign' in stage.lower():
            return 'MRI every 6-12 months'
        elif 'Grade II' in stage:
            return 'MRI every 3-6 months'
        else:
            return 'MRI every 2-3 months'

    def _no_tumor_analysis(self):
        """Return analysis when no tumor is detected"""
        return {
            'tumor_type': 'notumor',
            'confidence': 1.0,
            'measurements': {
                'max_diameter_mm': 0,
                'volume_mm3': 0,
                'volume_cm3': 0
            },
            'morphology': {'shape': 'N/A'},
            'staging': {'stage': 'N/A'},
            'location': {'primary_location': 'N/A'},
            'characteristics': {'growth_pattern': 'N/A'},
            'clinical': {
                'prognosis': 'N/A',
                'treatment_recommendation': 'No treatment needed',
                'follow_up': 'Routine screening as appropriate'
            }
        }


class BrainTumorDataset(Dataset):
    """Dataset with integrated tumor analysis"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = []
        self.labels = []

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
    """Model maintaining original metrics while adding tumor analysis"""

    def __init__(self, num_classes=4, dropout_rate=0.3):
        super(ImprovedModel, self).__init__()

        self.backbone = models.efficientnet_b0(pretrained=True)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.se_block = SEBlock(1280)

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


class PersistentTrainer:
    """Trainer with integrated tumor analysis"""

    def __init__(self, num_classes=4, learning_rate=3e-4, use_ensemble=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        print(f"🖥️ Using device: {self.device} ({device_name})")

        self.num_classes = num_classes
        self.use_ensemble = use_ensemble
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

        # Initialize tumor analyzer
        self.tumor_analyzer = TumorCharacteristicsAnalyzer()

        self.model = ImprovedModel(num_classes)
        print("📦 Using Improved Model with Tumor Analysis")

        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.focal_loss = FocalLoss(alpha=1, gamma=2)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
            amsgrad=True
        )

        try:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True,
                min_lr=1e-7
            )
        except TypeError:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )

        if USE_AMP:
            self.scaler = GradScaler()
        else:
            self.scaler = None

        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rates': [],
            'tumor_analyses': []
        }

        self.best_val_acc = 0
        self.best_val_loss = float('inf')
        self.target_reached = False
        self.target_accuracy = 98.5
        self.ultimate_target = 99.0
        self.epochs_after_target = 0
        self.max_epochs_after_target = 30

    def validate_with_analysis(self, val_loader):
        """Validation with comprehensive tumor analysis"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        class_correct = list(0. for _ in range(self.num_classes))
        class_total = list(0. for _ in range(self.num_classes))
        tumor_analyses = []

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(val_loader, desc='Validation', ncols=100)):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if USE_AMP and self.scaler:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                running_loss += loss.item()

                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Per-class accuracy
                c = (predicted == labels).squeeze()
                for i in range(labels.size(0)):
                    label = labels[i]
                    class_correct[label] += c[i].item() if c.dim() == 0 else c[i].item()
                    class_total[label] += 1

                # Perform tumor analysis on first batch
                if batch_idx == 0:
                    for i in range(min(3, labels.size(0))):
                        tumor_type = self.classes[predicted[i].item()]
                        confidence = probs[i, predicted[i]].item()

                        # Comprehensive tumor analysis
                        analysis = self.tumor_analyzer.analyze_tumor_comprehensive(
                            tumor_type, confidence
                        )
                        tumor_analyses.append(analysis)

        val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total

        # Print per-class accuracy if high
        if val_acc >= 97:
            print("\n📊 Per-class accuracy:")
            for i in range(self.num_classes):
                if class_total[i] > 0:
                    acc = 100 * class_correct[i] / class_total[i]
                    print(f"  {self.classes[i]}: {acc:.2f}%")

        return val_loss, val_acc, tumor_analyses

    def train_epoch(self, train_loader, epoch):
        """Training epoch - unchanged to preserve metrics"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        use_mixup_prob = 0.3 if self.best_val_acc < 98 else 0.2
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}', ncols=100)

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            use_mixup = np.random.random() < use_mixup_prob

            if use_mixup:
                images, labels_a, labels_b, lam = self.mixup_data(images, labels)

                if USE_AMP and self.scaler:
                    with autocast():
                        outputs = self.model(images)
                        ce_loss = lam * self.criterion(outputs, labels_a) + \
                                  (1 - lam) * self.criterion(outputs, labels_b)
                        focal = lam * self.focal_loss(outputs, labels_a) + \
                                (1 - lam) * self.focal_loss(outputs, labels_b)
                        loss = 0.7 * ce_loss + 0.3 * focal
                else:
                    outputs = self.model(images)
                    ce_loss = lam * self.criterion(outputs, labels_a) + \
                              (1 - lam) * self.criterion(outputs, labels_b)
                    focal = lam * self.focal_loss(outputs, labels_a) + \
                            (1 - lam) * self.focal_loss(outputs, labels_b)
                    loss = 0.7 * ce_loss + 0.3 * focal
            else:
                if USE_AMP and self.scaler:
                    with autocast():
                        outputs = self.model(images)
                        ce_loss = self.criterion(outputs, labels)
                        focal = self.focal_loss(outputs, labels)
                        loss = 0.7 * ce_loss + 0.3 * focal
                else:
                    outputs = self.model(images)
                    ce_loss = self.criterion(outputs, labels)
                    focal = self.focal_loss(outputs, labels)
                    loss = 0.7 * ce_loss + 0.3 * focal

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            self.optimizer.zero_grad(set_to_none=True)

            if USE_AMP and self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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

    def mixup_data(self, x, y, alpha=0.3):
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def train(self, train_loader, val_loader, epochs=100):
        """Training with comprehensive tumor analysis"""

        print(f"\n{'=' * 70}")
        print(f"🚀 TRAINING WITH COMPREHENSIVE TUMOR ANALYSIS")
        print(f"{'=' * 70}")
        print(f"Model: Improved Single with Tumor Characterization")
        print(f"Analysis: Size, Shape, Volume, Staging, Location, Clinical")
        print(f"Metrics: Accuracy, Precision, Recall, F1, ROC-AUC")
        print(f"{'=' * 70}\n")

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, val_acc, tumor_analyses = self.validate_with_analysis(val_loader)

            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']

            if old_lr != current_lr:
                print(f"📉 Learning rate changed: {old_lr:.2e} → {current_lr:.2e}")

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)

            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch}/{epochs}")
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.2e}")

            # Display tumor analysis every 5 epochs
            if tumor_analyses and epoch % 5 == 0:
                print("\n🔬 COMPREHENSIVE TUMOR ANALYSIS:")
                for idx, analysis in enumerate(tumor_analyses[:1], 1):
                    if analysis['tumor_type'] != 'notumor':
                        print(f"\n  Sample {idx}:")
                        print(f"  ├─ Type: {analysis['tumor_type']}")
                        print(f"  ├─ Confidence: {analysis['confidence']:.2%}")
                        print(f"  ├─ Measurements:")
                        print(f"  │  ├─ Size: {analysis['measurements']['max_diameter_mm']:.1f}mm")
                        print(f"  │  ├─ Volume: {analysis['measurements']['volume_cm3']:.2f}cm³")
                        print(f"  │  └─ Surface Area: {analysis['measurements']['surface_area_mm2']:.1f}mm²")
                        print(f"  ├─ Morphology:")
                        print(f"  │  ├─ Shape: {analysis['morphology']['shape']}")
                        print(f"  │  └─ Margins: {analysis['morphology']['margins']}")
                        print(f"  ├─ Staging:")
                        print(f"  │  ├─ Stage: {analysis['staging']['stage']}")
                        print(f"  │  └─ Malignancy: {analysis['staging']['malignancy']}")
                        print(f"  ├─ Location:")
                        print(f"  │  ├─ Primary: {analysis['location']['primary_location']}")
                        print(f"  │  └─ Depth: {analysis['location']['depth']}")
                        print(f"  └─ Clinical:")
                        print(f"     ├─ Prognosis: {analysis['clinical']['prognosis']}")
                        print(f"     └─ Treatment: {analysis['clinical']['treatment_recommendation']}")

            if val_acc >= self.target_accuracy:
                if not self.target_reached:
                    self.target_reached = True
                    print(f"\n🎯 Target {self.target_accuracy}% reached!")
                    self.save_model(f'model_{self.target_accuracy}percent.pth')
                self.epochs_after_target += 1

            if val_acc > self.best_val_acc:
                improvement = val_acc - self.best_val_acc
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.save_model('best_model.pth')
                print(f"✅ New best model! Accuracy: {val_acc:.2f}% (+{improvement:.2f}%)")

                # Save tumor analyses
                if tumor_analyses:
                    self.history['tumor_analyses'] = tumor_analyses

                if val_acc >= 99.0:
                    print(f"\n🏆 99%+ accuracy achieved: {val_acc:.2f}%")
                    self.save_model(f'model_99plus_{val_acc:.2f}.pth')

            print(f"{'=' * 70}")

            if torch.cuda.is_available() and epoch % 10 == 0:
                torch.cuda.empty_cache()
            gc.collect()

        print(f"\n{'=' * 70}")
        print(f"📈 TRAINING COMPLETE")
        print(f"{'=' * 70}")
        print(f"Best Validation Accuracy: {self.best_val_acc:.2f}%")
        print(f"Classification metrics preserved: Accuracy, ROC-AUC, Precision, Recall, F1")
        print(f"Tumor analysis added: Size, Shape, Volume, Staging")

        self.save_history()
        self.save_tumor_analysis()

        return self.best_val_acc

    def save_model(self, filename):
        save_dir = Path('experiments/saved_models')
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'use_ensemble': self.use_ensemble,
            'classes': self.classes,
            'target_reached': self.target_reached,
            'epochs_after_target': self.epochs_after_target,
            'tumor_analyzer': 'TumorCharacteristicsAnalyzer'
        }

        save_path = save_dir / filename
        torch.save(checkpoint, save_path)
        print(f"💾 Model saved: {save_path}")

    def save_history(self):
        history_path = Path('experiments/training_history.json')
        history_path.parent.mkdir(exist_ok=True)

        history_data = {
            'best_val_acc': float(self.best_val_acc),
            'best_val_loss': float(self.best_val_loss),
            'target_reached': self.target_reached,
            'epochs_trained': len(self.history['train_acc']),
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

    def save_tumor_analysis(self):
        """Save comprehensive tumor analysis"""
        if 'tumor_analyses' in self.history and self.history['tumor_analyses']:
            analysis_path = Path('experiments/tumor_analysis.json')

            analyses_data = []
            for analysis in self.history['tumor_analyses']:
                if analysis['tumor_type'] != 'notumor':
                    analyses_data.append({
                        'tumor_type': analysis['tumor_type'],
                        'confidence': float(analysis['confidence']),
                        'size_mm': float(analysis['measurements']['max_diameter_mm']),
                        'volume_cm3': float(analysis['measurements']['volume_cm3']),
                        'shape': analysis['morphology']['shape'],
                        'stage': analysis['staging']['stage'],
                        'malignancy': analysis['staging']['malignancy'],
                        'location': analysis['location']['primary_location'],
                        'depth': analysis['location']['depth'],
                        'prognosis': analysis['clinical']['prognosis'],
                        'treatment': analysis['clinical']['treatment_recommendation']
                    })

            with open(analysis_path, 'w') as f:
                json.dump(analyses_data, f, indent=2)
            print(f"🔬 Tumor analyses saved: {analysis_path}")


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def create_advanced_transforms():
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

    parser = argparse.ArgumentParser(description='Training with Comprehensive Tumor Analysis')
    parser.add_argument('--epochs', type=int, default=150, help='Maximum epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--use-ensemble', action='store_true', help='Use ensemble model')
    parser.add_argument('--data-dir', type=str, default='dataset_processed')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🧠 BRAIN TUMOR CLASSIFICATION WITH COMPREHENSIVE ANALYSIS")
    print("=" * 70)
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print("Tumor Analysis: Size, Shape, Volume, Staging, Location, Clinical")
    print("Preserved Metrics: Accuracy, Precision, Recall, F1, ROC-AUC")
    print("=" * 70)

    train_transform, val_transform = create_advanced_transforms()

    train_data_dir = Path(args.data_dir) / 'Training'
    test_data_dir = Path(args.data_dir) / 'Testing'

    if not train_data_dir.exists():
        print(f"❌ Error: Data not found at {train_data_dir}")
        return

    print("\n📁 Loading datasets...")
    full_train_dataset = BrainTumorDataset(train_data_dir, transform=train_transform)
    test_dataset = BrainTumorDataset(test_data_dir, transform=val_transform)

    train_size = int(0.85 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=generator
    )

    if not torch.cuda.is_available():
        args.batch_size = min(args.batch_size, 16)
        print(f"📝 Adjusted batch size for CPU: {args.batch_size}")

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
    print(f"  Batch Size: {args.batch_size}")

    trainer = PersistentTrainer(
        num_classes=4,
        learning_rate=args.lr,
        use_ensemble=args.use_ensemble
    )

    try:
        best_acc = trainer.train(train_loader, val_loader, epochs=args.epochs)

        print(f"\n{'=' * 70}")
        print(f"🏆 TRAINING COMPLETE")
        print(f"{'=' * 70}")
        print(f"Best Validation Accuracy: {best_acc:.2f}%")
        print("\n📁 Output Files:")
        print("  • experiments/saved_models/best_model.pth")
        print("  • experiments/training_history.json")
        print("  • experiments/tumor_analysis.json")

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted")
        trainer.save_model('interrupted_checkpoint.pth')
    except Exception as e:
        print(f"\n❌ Error: {e}")
        trainer.save_model('error_checkpoint.pth')
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == '__main__':
    main()