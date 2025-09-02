import numpy as np
import SimpleITK as sitk
from scipy import ndimage


class VolumeCalculator:
    def __init__(self, voxel_spacing=(1.0, 1.0, 1.0)):
        """
        Initialize volume calculator
        voxel_spacing: (x, y, z) spacing in mm
        """
        self.voxel_spacing = voxel_spacing

    def calculate_volume_from_mask(self, mask):
        """
        Calculate tumor volume from binary mask
        """
        # Count positive voxels
        tumor_voxels = np.sum(mask > 0)

        # Calculate voxel volume
        voxel_volume = np.prod(self.voxel_spacing)

        # Total volume in mm³
        tumor_volume = tumor_voxels * voxel_volume

        # Convert to cm³
        tumor_volume_cm3 = tumor_volume / 1000

        return {
            'volume_mm3': tumor_volume,
            'volume_cm3': tumor_volume_cm3,
            'voxel_count': tumor_voxels
        }

    def calculate_largest_diameter(self, mask):
        """
        Calculate the largest diameter of the tumor
        """
        if np.sum(mask) == 0:
            return 0

        # Find connected components
        labeled_mask, num_features = ndimage.label(mask)

        if num_features == 0:
            return 0

        # Find the largest component
        sizes = ndimage.sum(mask, labeled_mask, range(num_features + 1))
        max_label = sizes[1:].argmax() + 1
        largest_component = labeled_mask == max_label

        # Calculate distances
        distances = ndimage.distance_transform_edt(
            largest_component,
            sampling=self.voxel_spacing[:2]  # 2D for single slice
        )

        # Maximum diameter
        max_diameter = 2 * distances.max()

        return max_diameter

    def calculate_surface_area(self, mask):
        """
        Calculate tumor surface area
        """
        # Create boundary mask
        eroded = ndimage.binary_erosion(mask)
        boundary = mask & ~eroded

        # Count boundary voxels
        boundary_voxels = np.sum(boundary)

        # Approximate surface area
        voxel_surface = self.voxel_spacing[0] * self.voxel_spacing[1]
        surface_area = boundary_voxels * voxel_surface

        return surface_area

    def get_3d_measurements(self, mask_stack):
        """
        Calculate 3D measurements from stack of 2D masks
        """
        volume_info = self.calculate_volume_from_mask(mask_stack)

        # Calculate bounding box
        coords = np.argwhere(mask_stack > 0)
        if len(coords) > 0:
            min_coords = coords.min(axis=0)
            max_coords = coords.max(axis=0)
            dimensions = (max_coords - min_coords + 1) * self.voxel_spacing

            volume_info['dimensions_mm'] = dimensions
            volume_info['bbox'] = {
                'min': min_coords.tolist(),
                'max': max_coords.tolist()
            }

        return volume_info