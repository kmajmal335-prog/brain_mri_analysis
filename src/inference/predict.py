import torch
import numpy as np
from PIL import Image
import cv2


class BrainTumorPredictor:
    def __init__(self, classifier_path, segmenter_path, config):
        self.config = config
        self.device = torch.device(config.DEVICE)

        # Load models
        self.classifier = self._load_classifier(classifier_path)
        self.segmenter = self._load_segmenter(segmenter_path)

        # Initialize analyzers
        from ..evaluation.volume_calc import VolumeCalculator
        from ..evaluation.shape_features import ShapeAnalyzer
        from ..evaluation.staging_rules import TumorStaging

        self.volume_calc = VolumeCalculator()
        self.shape_analyzer = ShapeAnalyzer()
        self.staging = TumorStaging()

    def _load_classifier(self, path):
        from ..models.classifier import BrainTumorClassifier
        model = BrainTumorClassifier(
            num_classes=self.config.NUM_CLASSES,
            backbone=self.config.CLF_BACKBONE
        )
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def _load_segmenter(self, path):
        from ..models.segmenter import AttentionUNet
        model = AttentionUNet()
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def predict(self, image_path):
        """
        Complete prediction pipeline for a single image
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')

        # Prepare for classification
        clf_input = self._preprocess_for_classification(image)

        # Prepare for segmentation
        seg_input = self._preprocess_for_segmentation(image)

        results = {}

        with torch.no_grad():
            # Classification
            clf_output, attention = self.classifier(clf_input)
            probs = torch.softmax(clf_output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()

            results['tumor_detected'] = pred_class != 2  # notumor = index 2
            results['tumor_type'] = self.config.CLASS_NAMES[pred_class]
            results['confidence'] = probs[0, pred_class].item()
            results['all_probabilities'] = {
                self.config.CLASS_NAMES[i]: probs[0, i].item()
                for i in range(self.config.NUM_CLASSES)
            }

            # Segmentation if tumor detected
            if results['tumor_detected']:
                seg_output = self.segmenter(seg_input)
                mask = (seg_output[0, 0].cpu().numpy() > 0.5).astype(np.uint8)

                # Volume calculation
                volume_info = self.volume_calc.calculate_volume_from_mask(mask)
                results['volume'] = volume_info

                # Shape analysis
                shape_features = self.shape_analyzer.extract_shape_features(mask)
                results['shape_features'] = shape_features

                # Calculate diameter
                max_diameter = self.volume_calc.calculate_largest_diameter(mask)
                results['max_diameter_mm'] = max_diameter

                # Staging
                stage_info = self.staging.stage_tumor(
                    results['tumor_type'],
                    volume_info['volume_mm3'],
                    shape_features,
                    max_diameter
                )
                results['staging'] = stage_info

                # Treatment recommendations
                results['recommendations'] = self.staging.get_treatment_recommendation(
                    stage_info, results['tumor_type']
                )

                # Store mask for visualization
                results['segmentation_mask'] = mask
            else:
                results['volume'] = {'volume_mm3': 0, 'volume_cm3': 0}
                results['shape_features'] = {}
                results['staging'] = {'grade': 'N/A', 'malignancy': 'none'}
                results['recommendations'] = ['No tumor detected. Regular screening recommended.']

        return results

    def _preprocess_for_classification(self, image):
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize(self.config.IMG_SIZE_CLF),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        tensor = transform(image).unsqueeze(0).to(self.device)
        return tensor

    def _preprocess_for_segmentation(self, image):
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize(self.config.IMG_SIZE_SEG),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        tensor = transform(image).unsqueeze(0).to(self.device)
        return tensor