import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from metrics import print_metrics
from imageops import MinMaxNorm

def replace_filenames(folder_path, str_to_replace='', replace_with=''):
    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)
        new_file = file.replace(str_to_replace, replace_with)
        new_file_path = os.path.join(folder_path, new_file)
        os.rename(file_path, new_file_path)

        return None

def slices_to_stack(slice_path, output_path, bigtiff=False):
    # Gather all TIFF image paths
    img_files = sorted([f for f in os.listdir(slice_path) if f.lower().endswith(('.tif', '.tiff'))])

    # Check if there are any TIFF files
    if not img_files:
        raise ValueError("No TIFF images found in the specified folder.")

    # Load the first image to determine dimensions
    first_image_path = os.path.join(slice_path, img_files[0])
    first_image = imread(first_image_path)
    img_shape = first_image.shape  # (height, width)

    # Initialize an empty numpy array for the 3D volume (Z, Y, X)
    volume = np.zeros((len(img_files), img_shape[0], img_shape[1]), dtype=first_image.dtype)

    # Populate the volume array with each slice
    for i, filename in enumerate(img_files):
        img_path = os.path.join(slice_path, filename)
        volume[i, :, :] = imread(img_path)

    # Save the volume as a compressed .npz file
    imsave(output_path, volume, bigtiff=bigtiff, check_contrast=False)
    print(f"Volume saved as a multi-page TIFF file at {output_path}")

def extract_roi(image_dir, roi_dir, output_path, img_str='', roi_str=''):

    def add_string_to_filename(filename, string_to_add):
        # Split the filename and extension
        file_name, file_extension = os.path.splitext(filename)
        # Concatenate the parts with the string in between
        new_filename = f"{file_name}{string_to_add}{file_extension}"
        return new_filename

    img_files = os.listdir(image_dir)
    for img_file in img_files:
        img_path = os.path.join(image_dir, img_file)
        roi_file = os.path.join(roi_dir, img_file.replace(img_str, ''))
        roi_file = add_string_to_filename(roi_file, roi_str)
        img = imread(img_path)
        roi = imread(roi_file).astype(np.uint8)

        idx = np.nonzero(img)
        img_post_roi = np.zeros_like(img)
        for (m, n, l) in zip(*idx):
            if roi[m, n, l] > 0:
                img_post_roi[m, n, l] = img[m, n, l]

        file = os.path.join(output_path, img_file)
        imsave(file, img_post_roi.astype(img.dtype), bigtiff=False, check_contrast=False)

def plot_slices(gt_slice, seg_slice, slice_number, filename=''):
    """
    Plot and compare a slice from the ground truth and segmentation masks for double-checking metric output.

    Args:
    gt_slice (numpy array): A slice from the ground truth mask.
    seg_slice (numpy array): A corresponding slice from the segmentation mask.
    slice_number (int): The index of the slice.
    filename (str): The name of the file being processed (for title).
    """
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(gt_slice, cmap='gray')
    plt.title(f'Ground Truth Slice - {filename} - Slice {slice_number}')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(seg_slice, cmap='gray')
    plt.title(f'Segmentation Slice - {filename} - Slice {slice_number}')
    plt.axis('off')

    plt.show()

def segmentation_performance(gt_path, seg_path, seg_prefix='VANGAN_'):
    """Process and compare segmented and ground truth volumes in specified folders."""
    seg_files = sorted(os.listdir(seg_path))

    for seg_file in seg_files:
        seg_file_path = os.path.join(seg_path, seg_file)
        gt_file = seg_file.replace(seg_prefix, '')
        gt_file_path = os.path.join(gt_path, gt_file)

        # Check if ground truth file exists
        if not os.path.exists(gt_file_path):
            print(f"Warning: Ground truth file '{gt_file}' does not exist for '{seg_file}'. Skipping...")
            continue

        # Load segmentation and ground truth volumes
        seg_img = imread(seg_file_path)
        gt_img = imread(gt_file_path)

        # Ensure images are normalised
        op = MinMaxNorm()
        seg_img = op(seg_img)
        gt_img = op(gt_img)

        # Retrieve filename for metric logging
        filename = os.path.splitext(gt_file)[0]

        # Compute metrics and plot results
        print_metrics(filename, gt_img, seg_img)
        slice_number = int(0.5*gt_img.shape[0])
        plot_slices(gt_img[slice_number,], seg_img[slice_number,], slice_number)