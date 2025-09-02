import numpy as np
import cv2
from skimage import measure
from scipy import ndimage


class ShapeAnalyzer:
    def __init__(self):
        self.features = {}

    def extract_shape_features(self, mask):
        """
        Extract shape-based features from tumor mask
        """
        if np.sum(mask) == 0:
            return self._empty_features()

        # Convert to binary
        binary_mask = (mask > 0).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return self._empty_features()

        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Basic measurements
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)

        # Circularity (1 = perfect circle, <1 = irregular)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

        # Fit ellipse
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            (center, (MA, ma), angle) = ellipse

            # Eccentricity
            eccentricity = np.sqrt(1 - (ma / MA) ** 2) if MA > 0 else 0

            # Aspect ratio
            aspect_ratio = MA / ma if ma > 0 else 1
        else:
            eccentricity = 0
            aspect_ratio = 1
            MA, ma = 0, 0

        # Solidity (ratio of contour area to convex hull area)
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        # Irregularity index (custom metric)
        irregularity = self._calculate_irregularity(largest_contour, binary_mask)

        # Compactness
        compactness = area / (perimeter ** 2) if perimeter > 0 else 0

        features = {
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity,
            'eccentricity': eccentricity,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity,
            'irregularity': irregularity,
            'compactness': compactness,
            'major_axis': MA,
            'minor_axis': ma
        }

        return features

    def _calculate_irregularity(self, contour, mask):
        """
        Calculate shape irregularity (0 = regular, 1 = highly irregular)
        """
        # Calculate distance from centroid to boundary
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return 0

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Calculate distances from centroid to contour points
        distances = []
        for point in contour:
            x, y = point[0]
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            distances.append(dist)

        if len(distances) == 0:
            return 0

        distances = np.array(distances)
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)

        # Irregularity as coefficient of variation
        irregularity = std_dist / mean_dist if mean_dist > 0 else 0

        return min(irregularity, 1.0)  # Cap at 1.0

    def _empty_features(self):
        return {
            'area': 0,
            'perimeter': 0,
            'circularity': 0,
            'eccentricity': 0,
            'aspect_ratio': 1,
            'solidity': 0,
            'irregularity': 0,
            'compactness': 0,
            'major_axis': 0,
            'minor_axis': 0
        }

    def classify_shape_malignancy(self, features):
        """
        Classify tumor as benign or malignant based on shape features
        """
        score = 0

        # Irregularity scoring
        if features['irregularity'] > 0.7:
            score += 3
        elif features['irregularity'] > 0.5:
            score += 2
        elif features['irregularity'] > 0.3:
            score += 1

        # Circularity scoring (irregular shapes more likely malignant)
        if features['circularity'] < 0.5:
            score += 2
        elif features['circularity'] < 0.7:
            score += 1

        # Solidity scoring
        if features['solidity'] < 0.7:
            score += 2
        elif features['solidity'] < 0.85:
            score += 1

        # Eccentricity scoring
        if features['eccentricity'] > 0.8:
            score += 1

        # Classification
        if score >= 5:
            return 'likely_malignant', score / 9.0
        elif score >= 3:
            return 'suspicious', score / 9.0
        else:
            return 'likely_benign', score / 9.0