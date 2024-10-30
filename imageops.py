import os
import cv2
import tifffile
import numpy as np
from skimage import io

class Operation:
    def set_global_stats(self, stats):
        pass

    @property
    def needs_stats(self):
        return []

    def __call__(self, img):
        raise NotImplementedError

class ApplyGammaCorrection(Operation):
    def __init__(self, gamma=2.0):
        self.gamma = gamma

    def __call__(self, img):
        img = img.astype(np.float32)
        max_val = np.max(img)
        return ((img / max_val) ** self.gamma) * max_val if max_val > 0 else img

class ApplyCLAHE(Operation):
    def __init__(self, clip_limit=2.0, grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.grid_size = grid_size
        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.grid_size)

    def __call__(self, img):
        return self.clahe.apply(img.astype(np.uint8))

class MinMaxNorm(Operation):
    def __init__(self):
        self.global_min = None
        self.global_max = None

    def set_global_stats(self, stats):
        self.global_min = stats.get('min')
        self.global_max = stats.get('max')

    @property
    def needs_stats(self):
        return ['min', 'max']

    def __call__(self, img):
        if self.global_min is not None and self.global_max is not None:
            return (img - self.global_min) / (self.global_max - self.global_min)
        else:
            img_min = np.min(img)
            img_max = np.max(img)
            return (img - img_min) / (img_max - img_min)

class ZScoreNorm(Operation):
    def __init__(self):
        self.global_mean = None
        self.global_std = None

    def set_global_stats(self, stats):
        self.global_mean = stats.get('mean')
        self.global_std = stats.get('std')

    @property
    def needs_stats(self):
        return ['mean', 'std']

    def __call__(self, img):
        if self.global_mean is not None and self.global_std is not None:
            return (img - self.global_mean) / self.global_std if self.global_std > 0 else img - self.global_mean
        else:
            mean = np.mean(img)
            std = np.std(img)
            return (img - mean) / std if std > 0 else img - mean

def compute_global_statistics(input_path, stats=None, sample_rate=0.01):
    """
    Computes global statistics over an image by sampling slices.

    Args:
        input_path (str): Path to the input image file.
        stats (list of str): List of statistics to compute -> 'min', 'max', 'mean', 'std'
        sample_rate (float): Proportion of slices to sample.

    Returns:
        dict: Dictionary of computed statistics.
    """
    if stats is None:
        stats = ['min', 'max', 'mean', 'std']
    with tifffile.TiffFile(input_path) as tif:
        total_slices = len(tif.pages)
        interval = max(1, int(1 / sample_rate))
        sampled_values = []
        for idx in range(0, total_slices, interval):
            img_slice = tif.pages[idx].asarray()
            sampled_values.extend(img_slice.ravel()[::interval])
        sampled_values = np.array(sampled_values)
        result = {}
        if 'min' in stats:
            result['min'] = np.min(sampled_values)
        if 'max' in stats:
            result['max'] = np.max(sampled_values)
        if 'mean' in stats:
            result['mean'] = np.mean(sampled_values)
        if 'std' in stats:
            result['std'] = np.std(sampled_values)
        return result

def process_image(input_path, output_path, operations, per_slice=False, precision=np.float32, sample_rate=0.01):
    """
    Processes an image by applying operations.

    Args:
        input_path (str): Path to the input image file.
        output_path (str): Path to save the processed image.
        operations (list of Operation instances): List of operations to apply.
        per_slice (bool): Whether to apply operations per slice.
        precision (dtype): Desired data type for the output image.
        sample_rate (float): Sample rate for estimating global statistics.
    """
    # Determine if image is large (needs to be processed per slice)
    img_info = os.stat(input_path)
    large_image = img_info.st_size > 1e9 or per_slice  # Threshold size can be adjusted

    # Determine which statistics are needed
    stats_needed = set()
    for op in operations:
        stats_needed.update(op.needs_stats)

    if stats_needed:
        # Compute global statistics
        global_stats = compute_global_statistics(input_path, stats=list(stats_needed), sample_rate=sample_rate)
        # Set global statistics in operations
        for op in operations:
            op.set_global_stats(global_stats)

    if large_image:
        # Process per slice
        with tifffile.TiffFile(input_path) as tif:
            with tifffile.TiffWriter(output_path, bigtiff=True) as tif_writer:
                for page in tif.pages:
                    img_slice = page.asarray()
                    # Apply operations to the slice
                    for op in operations:
                        img_slice = op(img_slice)
                    # Save the processed slice
                    tif_writer.write(img_slice.astype(precision))
    else:
        # Load the entire image into memory
        img = io.imread(input_path)
        if per_slice and img.ndim == 3:
            # Apply operations per slice
            for i in range(img.shape[0]):
                img_slice = img[i]
                for op in operations:
                    img_slice = op(img_slice)
                img[i] = img_slice
        else:
            # Apply operations to the whole image
            for op in operations:
                img = op(img)
        # Save the processed image
        tifffile.imwrite(output_path, img.astype(precision))

def process_image_folder(input_folder, output_folder, operations, per_slice=False, precision=np.float32, sample_rate=1.0):
    """
    Processes all images in a folder by applying operations.

    Args:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder to save the processed images.
        operations (list of Operation instances): List of operations to apply.
        per_slice (bool): Whether to apply operations per slice.
        precision (dtype): Desired data type for the output images.
        sample_rate (float): Sample rate for estimating global statistics.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.tif', '.tiff'))]
    for image_file in image_files:
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)
        process_image(input_path, output_path, operations, per_slice=per_slice, precision=precision, sample_rate=sample_rate)
        print(f"Processed and saved: {output_path}")

def intensity_projection(image, projection='maximum', axis=0):
    """
    Perform a maximum intensity projection along a specified axis.

    Args:
        image (np.ndarray): Path to the 3D image stack file.
        projection (str): Type of intensity projection to apply.
        axis (int): Axis along which to compute the projection.

    Returns:
        np.ndarray: The maximum intensity projection image.
    """
    if projection == 'maximum':
        return np.max(image, axis=axis)
    elif projection == 'std':
        return np.std(image, axis=axis)
