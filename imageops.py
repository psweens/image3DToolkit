import cv2
import tifffile
import numpy as np
from skimage import io

def apply_gamma_correction(img_slice, gamma=2.0):
    """
    Apply gamma correction to an image slice.

    Args:
        img_slice (np.ndarray): 2D image slice.
        gamma (float): Gamma value for correction.

    Returns:
        np.ndarray: Gamma-corrected image slice.
    """
    img_slice = img_slice.astype(np.float32)
    max_val = np.max(img_slice)
    return ((img_slice / max_val) ** gamma) * max_val if max_val > 0 else img_slice


def apply_clahe(img_slice, clip_limit=2.0, grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image slice.

    Args:
        img_slice (np.ndarray): 2D image slice.
        clip_limit (float): Threshold for contrast limiting.
        grid_size (tuple): Grid size for histogram equalization.

    Returns:
        np.ndarray: CLAHE-enhanced image slice.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(img_slice.astype(np.uint8))

def minmax_norm(img_slice):
    """
    Apply min/max normalization to an image slice.

    Args:
        img_slice (np.ndarray): 2D image slice.

    Returns:
        np.ndarray: Z-score normalized image slice.
    """
    img_max = np.max(img_slice)
    img_min = np.min(img_slice)
    return (img_slice - img_min) / (img_max - img_min)

def z_score_norm(img_slice):
    """
    Apply Z-score normalization to an image slice.

    Args:
        img_slice (np.ndarray): 2D image slice.

    Returns:
        np.ndarray: Z-score normalized image slice.
    """
    mean = np.mean(img_slice)
    std = np.std(img_slice)
    return (img_slice - mean) / std if std > 0 else img_slice - mean

def maximum_intensity_projection(image_path, axis=0):
    """Perform a maximum intensity projection along a specified axis."""
    img = io.imread(image_path)
    mip = np.max(img, axis=axis)
    return mip

def get_normalisation_function(normalise):
    """
    Helper method to retrieve the normalization function based on the method specified.

    Args:
        normalise (str): Normalization method ('minmax' or 'zscore').

    Returns:
        function: Corresponding normalization function.
    """
    if normalise == 'minmax':
        return minmax_norm
    elif normalise == 'zscore':
        return z_score_norm
    return None

def load_image_volume(file_path, normalise='', load_in_chunks=False, operations=None, apply_globally=False):
    """
    Load a 3D image stack from a TIFF file, with options for normalization and processing large files.

    Args:
        file_path (str): Path to the 3D image stack file.
        normalise (str, optional): Global normalization method ('minmax' or 'zscore').
        load_in_chunks (bool, optional): Whether to load the image in chunks.
        operations (list of str, optional): List of slice-wise operations to apply per slice (e.g., 'gamma', 'clahe').
        apply_globally (bool, optional): Whether to apply normalization globally or slice-wise (default is False for slice-wise).

    Returns:
        np.ndarray or generator: Loaded and processed 3D image stack or generator of slices.
    """
    operations = operations or []
    normalisation_func = get_normalisation_function(normalise)

    if not load_in_chunks:
        image_vol = io.imread(file_path)

        # Apply global normalization before slice-wise operations, if specified
        if normalisation_func and apply_globally:
            image_vol = normalisation_func(image_vol)

        # Apply slice-wise operations
        image_vol = apply_slice_operations(image_vol, operations)

        # Apply global normalization slice-wise, if not applied globally
        if normalisation_func and not apply_globally:
            image_vol = apply_slice_normalization(image_vol, normalisation_func)

        return image_vol
    else:
        return load_in_chunks(file_path, normalisation_func, operations, apply_globally)

def _load_in_chunks(file_path, normalisation_func, operations, apply_globally):
    """
    Helper method for loading images in chunks (generator).
    """
    with tifffile.TiffFile(file_path) as tif:
        num_slices = len(tif.pages)

        if normalisation_func and apply_globally:
            global_stats = _estimate_global_statistics(file_path, normalisation_func)

        def slice_generator():
            for page in tif.pages:
                img_slice = page.asarray()

                # Apply slice-wise operations
                img_slice = apply_operations_to_slice(img_slice, operations)

                # Apply global normalization slice-wise, if specified
                if normalisation_func and not apply_globally:
                    img_slice = normalisation_func(img_slice)

                yield img_slice

        return slice_generator()

def apply_slice_operations(image_vol, operations):
    """
    Apply operations to each slice in a 3D volume (for slice-wise operations only).

    Args:
        image_vol (np.ndarray): 3D image stack.
        operations (list of str): List of slice-wise operations to apply.

    Returns:
        np.ndarray: Processed image stack.
    """
    for i in range(image_vol.shape[0]):
        image_vol[i] = apply_operations_to_slice(image_vol[i], operations)
    return image_vol

def apply_slice_normalization(image_vol, normalisation_func):
    """
    Apply normalization slice-by-slice to a 3D volume.

    Args:
        image_vol (np.ndarray): 3D image stack.
        normalisation_func (function): Normalization function to apply.

    Returns:
        np.ndarray: Normalized image stack.
    """
    for i in range(image_vol.shape[0]):
        image_vol[i] = normalisation_func(image_vol[i])
    return image_vol

def apply_operations_to_slice(img_slice, operations):
    """
    Apply specified operations to a single image slice.

    Args:
        img_slice (np.ndarray): 2D image slice.
        operations (list of str): List of operations to apply to each slice.

    Returns:
        np.ndarray: Processed image slice.
    """
    for op in operations:
        if op == 'gamma':
            img_slice = apply_gamma_correction(img_slice)
        elif op == 'clahe':
            img_slice = apply_clahe(img_slice)
        elif op == 'zscore_slice':
            img_slice = z_score_norm(img_slice)
    return img_slice

def _estimate_global_statistics(file_path, normalisation_func, sample_rate=0.01):
    """
    Estimate global statistics for normalization by sampling slices.

    Args:
        file_path (str): Path to the image stack file.
        normalisation_func (function): Normalization function.
        sample_rate (float): Proportion of slices to sample (default 1%).

    Returns:
        tuple: Global statistics needed for normalization.
    """
    sampled_values = _sample_slices(file_path, sample_rate)
    if normalisation_func == minmax_norm:
        return np.min(sampled_values), np.max(sampled_values)
    elif normalisation_func == z_score_norm:
        return np.mean(sampled_values), np.std(sampled_values)
    return None

def _sample_slices(file_path, sample_rate):
    """
    Sample slices from a large 3D image for estimating statistics.

    Args:
        file_path (str): Path to the image stack file.
        sample_rate (float): Proportion of slices to sample (default 1%).

    Returns:
        np.ndarray: Sampled values from the slices.
    """
    sampled_values = []
    with tifffile.TiffFile(file_path) as tif:
        total_slices = len(tif.pages)
        interval = max(1, int(1 / sample_rate))
        for idx in range(0, total_slices, interval):
            img_slice = tif.pages[idx].asarray()
            sampled_values.extend(img_slice.ravel()[::interval])
    return np.array(sampled_values)


def save_large_image(slice_generator, output_file, precision=np.float32):
    """
    Save slices from a generator to a TIFF file.

    Args:
        slice_generator (generator): Generator yielding image slices.
        output_file (str): Path to the output TIFF file.
        precision (np.dtype, optional): Data type for the output image.
    """
    with tifffile.TiffWriter(output_file, bigtiff=True) as tif_writer:
        for img_slice in slice_generator:
            tif_writer.write(img_slice.astype(precision))
