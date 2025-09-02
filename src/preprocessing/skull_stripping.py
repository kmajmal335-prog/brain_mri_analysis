import cv2
import numpy as np
from scipy import ndimage
from skimage import morphology


class SkullStripper:
    """Remove skull and non-brain regions from MRI"""

    def __init__(self):
        self.kernel_size = 5

    def strip_skull(self, image):
        """
        Remove skull from brain MRI image
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold using Otsu's method
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find largest connected component (brain)
        labels = morphology.label(closed)
        props = morphology.regionprops(labels)

        if props:
            largest_cc_idx = np.argmax([p.area for p in props])
            brain_mask = labels == props[largest_cc_idx].label

            # Fill holes
            brain_mask = ndimage.binary_fill_holes(brain_mask).astype(np.uint8)

            # Apply mask to original image
            if len(image.shape) == 3:
                result = image.copy()
                for i in range(3):
                    result[:, :, i] = result[:, :, i] * brain_mask
            else:
                result = image * brain_mask

            return result, brain_mask

        return image, np.ones_like(gray)