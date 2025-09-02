import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


class Visualizer:
    def __init__(self, config):
        self.config = config

    def visualize_prediction(self, image, results, save_path=None):
        """
        Create comprehensive visualization of prediction results
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original MRI')
        axes[0, 0].axis('off')

        # Segmentation overlay
        if 'segmentation_mask' in results and results['segmentation_mask'] is not None:
            overlay = self._create_overlay(image, results['segmentation_mask'])
            axes[0, 1].imshow(overlay)
            axes[0, 1].set_title('Tumor Segmentation')
        else:
            axes[0, 1].imshow(image)
            axes[0, 1].set_title('No Tumor Detected')
        axes[0, 1].axis('off')

        # Class probabilities
        if 'all_probabilities' in results:
            classes = list(results['all_probabilities'].keys())
            probs = list(results['all_probabilities'].values())
            axes[0, 2].bar(classes, probs)
            axes[0, 2].set_title('Classification Probabilities')
            axes[0, 2].set_ylim([0, 1])
            axes[0, 2].tick_params(axis='x', rotation=45)

        # Shape features
        if 'shape_features' in results and results['shape_features']:
            features = results['shape_features']
            feature_text = f"Circularity: {features.get('circularity', 0):.3f}\n"
            feature_text += f"Irregularity: {features.get('irregularity', 0):.3f}\n"
            feature_text += f"Solidity: {features.get('solidity', 0):.3f}\n"
            feature_text += f"Eccentricity: {features.get('eccentricity', 0):.3f}"
            axes[1, 0].text(0.1, 0.5, feature_text, fontsize=12, verticalalignment='center')
            axes[1, 0].set_title('Shape Analysis')
        axes[1, 0].axis('off')

        # Volume and staging info
        info_text = f"Tumor Type: {results.get('tumor_type', 'N/A')}\n"
        info_text += f"Confidence: {results.get('confidence', 0):.2%}\n"
        if 'volume' in results:
            info_text += f"Volume: {results['volume'].get('volume_cm3', 0):.2f} cm³\n"
        if 'staging' in results:
            info_text += f"Grade: {results['staging'].get('grade', 'N/A')}\n"
            info_text += f"Malignancy: {results['staging'].get('malignancy', 'N/A')}"

        axes[1, 1].text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center')
        axes[1, 1].set_title('Clinical Information')
        axes[1, 1].axis('off')

        # Recommendations
        if 'recommendations' in results:
            rec_text = '\n'.join([f"• {r}" for r in results['recommendations'][:3]])
            axes[1, 2].text(0.1, 0.5, rec_text, fontsize=10,
                            verticalalignment='center', wrap=True)
        axes[1, 2].set_title('Recommendations')
        axes[1, 2].axis('off')

        plt.suptitle('Brain Tumor Analysis Report', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def _create_overlay(self, image, mask):
        """
        Create overlay of segmentation mask on original image
        """
        # Resize mask to match image
        if isinstance(image, np.ndarray):
            img_array = image
        else:
            img_array = np.array(image)

        mask_resized = cv2.resize(mask, (img_array.shape[1], img_array.shape[0]))

        # Create colored mask
        colored_mask = np.zeros_like(img_array)
        colored_mask[:, :, 0] = mask_resized * 255  # Red channel

        # Blend with original
        overlay = cv2.addWeighted(img_array, 0.7, colored_mask, 0.3, 0)

        return overlay

    def generate_gradcam(self, model, image_tensor, target_layer_name='base_model'):
        """
        Generate Grad-CAM visualization
        """
        # Get target layer
        target_layer = getattr(model, target_layer_name)

        # Initialize Grad-CAM
        cam = GradCAM(model=model, target_layers=[target_layer])

        # Generate CAM
        grayscale_cam = cam(input_tensor=image_tensor)

        return grayscale_cam