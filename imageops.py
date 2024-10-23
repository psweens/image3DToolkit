import numpy as np
import cv2
from skimage import io

class ImageOps:
    """
    A class encapsulating operations for 3D image stacks, including enhancements and normalisation.
    """

    def load_image_volume(self, file_path, normalise=''):
        """
        Load a 3D image stack from a TIFF file.

        Args:
            file_path (str): Path to the 3D image stack file.
            normalise (str, optional): Normalise the image stack.

        Returns:
            np.ndarray: Loaded 3D image stack.
        """
        image_vol = io.imread(file_path)
        if normalise == 'minmax': return self.min_max_norm(image_vol)
        elif normalise == 'zscore': return self.z_score_norm(image_vol)
        else: return image_vol

    @staticmethod
    def min_max_norm(image):
        """
        Normalise a 3D image array to a specific range.

        Args:
            image (np.ndarray): 3D NumPy array to normalise.

        Returns:
            np.ndarray: Normalised 3D image array.
        """
        img_min = np.min(image)
        img_max = np.max(image)
        if (img_max - img_min) == 0.:
            raise ValueError("Cannot perform min-max normalization when max and min are equal.")
        return (image - img_min) / (img_max - img_min)

    @staticmethod
    def z_score_norm(image):
        """
        Standardise a 3D image array (Z-score normalisation).

        Args:
            image (np.ndarray): 3D NumPy array to normalise.

        Returns:
            np.ndarray: Normalised 3D image array.
        """
        img_std = np.std(image)
        img_mean = np.mean(image)
        if img_std == 0.:
            raise ValueError("Cannot standardise image when image as STD of zero.")
        return (image - img_mean) / img_std

    @staticmethod
    def apply_clahe_3d(image, clip_limit=2.0, grid_size=(8, 8), axis=0):
        """
        Apply CLAHE to each slice of a 3D image array along the specified axis.

        Args:
            image (np.ndarray): 3D NumPy array representing the image stack.
            clip_limit (float): Threshold for contrast limiting.
            grid_size (tuple): Size of grid for the CLAHE algorithm.
            axis (int): Axis along which to apply CLAHE (0, 1, or 2).

        Returns:
            np.ndarray: CLAHE-enhanced image stack.
        """
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

        # Move the specified axis to the first position
        moved_image = np.moveaxis(image, axis, 0)

        # Initialize an empty array for the enhanced image stack
        enhanced_stack = np.zeros_like(moved_image, dtype=image.dtype)

        # Apply CLAHE to each 2D slice along the specified axis
        for i in range(moved_image.shape[0]):
            enhanced_stack[i, :, :] = clahe.apply(moved_image[i, :, :].astype(np.uint8))

        # Move the axis back to its original position
        enhanced_stack = np.moveaxis(enhanced_stack, 0, axis)

        return enhanced_stack
