import os
import math
import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt
import imageops as imo

def overlay_segmentation_on_image(confocal_img, segmentation_mask, alpha=0.5):
    """Overlay the instance segmentation mask onto the confocal image."""

    # Blend images with alpha transparency for the mask
    overlay_img = (1 - alpha) * confocal_img + alpha * segmentation_mask
    overlay_img = np.clip(overlay_img, 0, 255).astype(np.uint8)
    return overlay_img

def add_label_to_image(image, label, font_size=10):
    """Add a label to the bottom of an image using matplotlib for text rendering."""
    fig, ax = plt.subplots()
    ax.text(0.5, -0.1, label, ha='center', va='bottom', transform=ax.transAxes, fontsize=font_size, color='black')
    ax.imshow(image)
    ax.axis('off')

    fig.canvas.draw()
    labeled_image = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)

    return labeled_image

def calculate_grid_size(num_images):
    """Calculate the appropriate grid size (rows, cols) based on the number of images."""
    cols = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)
    return rows, cols

def create_image_panel(image_dir, mask_dir, output_filename, max_panels=10, is_3d=False, projection_axis=0,
                            overlay_mask=True):
    """Overlay segmentation masks on images with a maximum number of panels, optional 3D projection, and mask overlay."""
    original_images = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])
    segmentation_masks = sorted([f for f in os.listdir(mask_dir) if f.endswith('.tif')])

    if not original_images:
        print("No images found in the original folder.")
        return

    # Process images in batches of max_panels
    batch_count = math.ceil(len(original_images) / max_panels)

    for batch_index in range(batch_count):
        start_index = batch_index * max_panels
        end_index = start_index + max_panels
        current_original_images = original_images[start_index:end_index]
        current_segmentation_masks = segmentation_masks[start_index:end_index]

        num_images = len(current_original_images)
        rows, cols = calculate_grid_size(num_images)
        panels = []

        for original_image_name, mask_image_name in zip(current_original_images, current_segmentation_masks):
            original_image_path = os.path.join(image_dir, original_image_name)
            mask_image_path = os.path.join(mask_dir, mask_image_name)

            if is_3d:
                original_img = imo.intensity_projection(original_image_path, axis=projection_axis)
                mask_img = imo.intensity_projection(mask_image_path, axis=projection_axis) if overlay_mask else None
            else:
                original_img = io.imread(original_image_path)
                mask_img = io.imread(mask_image_path) if overlay_mask else None

            # Convert images to RGBA if they are grayscale or RGB
            if original_img.ndim == 2:  # Grayscale to RGBA
                original_img = color.gray2rgb(original_img)
            if original_img.shape[-1] == 3:  # RGB to RGBA
                original_img = np.dstack([original_img, np.full(original_img.shape[:2], 255, dtype=np.uint8)])

            if mask_img is not None:
                if mask_img.ndim == 2:  # Grayscale to RGBA
                    mask_img = color.gray2rgb(mask_img)
                if mask_img.shape[-1] == 3:  # RGB to RGBA
                    mask_img = np.dstack([mask_img, np.full(mask_img.shape[:2], 128, dtype=np.uint8)])  # Example: half transparency

            if overlay_mask and mask_img is not None:
                overlay_img = overlay_segmentation_on_image(original_img, mask_img)
            else:
                overlay_img = original_img

            # Add the filename as a label
            labeled_overlay_img = add_label_to_image(overlay_img, original_image_name)
            panels.append(labeled_overlay_img)

        # Determine dimensions for the combined image
        panel_height, panel_width, _ = panels[0].shape
        combined_image = np.ones((rows * panel_height, cols * panel_width, 4), dtype=np.uint8) * 255  # RGBA

        for index, panel in enumerate(panels):
            row_idx = (index // cols) * panel_height
            col_idx = (index % cols) * panel_width
            combined_image[row_idx:row_idx + panel_height, col_idx:col_idx + panel_width] = panel

        # Save each batch with a unique filename
        batch_filename = f"{output_filename}_batch_{batch_index + 1}.png"
        io.imsave(batch_filename, combined_image)
        print(f"Saved combined image to {batch_filename}")
