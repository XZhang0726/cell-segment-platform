"""
Segment Anything integration, exposed under the legacy CellSAM name.

Use Meta's Segment Anything Model (SAM) to generate cell-image masks.
This wrapper uses SamAutomaticMaskGenerator, not a dedicated CellSAM checkpoint.

Notes:
- SAM inference runs in the current environment with the required dependencies installed.
- Automatic point prompts guide mask generation.
- Download a SAM model checkpoint before use.
"""
import time
import numpy as np
from loguru import logger
from pathlib import Path
from typing import Optional

# Defer imports to reduce startup overhead.
_sam_model_cache = {}


def cellsam_segment(
    image: np.ndarray,
    model_type: str = "vit_b",
    use_gpu: bool = False,
    points_per_side: int = 32,
    progress_bar=None
) -> np.ndarray:
    """
    Generate cell-image masks using SAM through the legacy cellsam interface.

    Run inference directly in the current environment.

    Args:
        image: Input RGB image (H, W, C) or grayscale image (H, W).
        model_type: SAM model type: "vit_b", "vit_l", or "vit_h".
        use_gpu: Enable GPU acceleration when available.
        points_per_side: Number of automatically generated prompt points along each image side.
        progress_bar: Streamlit progress bar object.

    Returns:
        Segmentation mask with a unique label for each cell.

    Raises:
        RuntimeError: If inference fails.
    """
    start_time = time.time()

    # Check the image format.
    original_shape = image.shape
    if image.ndim == 2:
        # Convert grayscale to RGB.
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[2] == 4:
        # Convert RGBA to RGB.
        image = image[:, :, :3]
    elif image.ndim == 3 and image.shape[2] != 3:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    logger.info(f"CellSAM segmentation: image_shape={original_shape}, model_type={model_type}")

    try:
        # Import the required libraries.
        import torch
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

        # Update the progress bar.
        if progress_bar is not None:
            progress_bar.progress(0.1)

        # Select the device.
        device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Load the model, reusing the cache when available.
        model_key = f"{model_type}_{device}"
        if model_key not in _sam_model_cache:
            logger.info(f"Loading SAM model: {model_type}")
            model_load_start = time.time()

            # Determine the model checkpoint path.
            project_root = Path(__file__).parent.parent.parent.parent
            checkpoint_dir = project_root / "models" / "sam"

            # Map model types to checkpoint filenames.
            checkpoint_files = {
                "vit_b": "sam_vit_b_01ec64.pth",
                "vit_l": "sam_vit_l_0b3195.pth",
                "vit_h": "sam_vit_h_4b8939.pth"
            }

            if model_type not in checkpoint_files:
                raise ValueError(f"Unsupported model type: {model_type}")

            checkpoint_path = checkpoint_dir / checkpoint_files[model_type]

            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Model checkpoint not found at {checkpoint_path}\n"
                    f"Please download the model from: https://github.com/facebookresearch/segment-anything#model-checkpoints"
                )

            # Load the SAM model.
            sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
            sam.to(device=device)

            _sam_model_cache[model_key] = sam

            model_load_time = time.time() - model_load_start
            logger.info(f"Model loaded in {model_load_time:.2f}s")
        else:
            sam = _sam_model_cache[model_key]
            logger.info(f"Using cached model: {model_type}")

        # Update the progress bar.
        if progress_bar is not None:
            progress_bar.progress(0.3)

        # Create the automatic mask generator.
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Minimum region area for removing small fragments.
        )

        # Update the progress bar.
        if progress_bar is not None:
            progress_bar.progress(0.5)

        # Run inference.
        logger.info("Starting inference...")
        inference_start = time.time()

        # SAM requires an RGB image with uint8 values.
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

        masks = mask_generator.generate(image)

        inference_time = time.time() - inference_start
        logger.info(f"Inference completed in {inference_time:.2f}s, generated {len(masks)} masks")

        # Update the progress bar.
        if progress_bar is not None:
            progress_bar.progress(0.8)

        # Postprocess by combining masks into an instance label image.
        logger.info("Post-processing masks...")
        postprocess_start = time.time()

        h, w = image.shape[:2]
        image_area = h * w
        instance_map = np.zeros((h, w), dtype=np.int32)

        # Filter masks by area to remove oversized regions and small fragments.
        filtered_masks = []
        for mask_data in masks:
            mask = mask_data['segmentation']
            mask_area = np.sum(mask)
            area_ratio = mask_area / image_area

            # Filtering criteria:
            # 1. Masks must cover less than 50% of the image to exclude background-like regions.
            # 2. SAM's min_mask_region_area setting handles regions smaller than 100 pixels.
            if area_ratio < 0.5:
                filtered_masks.append(mask_data)
            else:
                logger.debug(f"Filtered out large mask: area={mask_area}, ratio={area_ratio:.2%}")

        logger.info(f"Filtered masks: {len(masks)} -> {len(filtered_masks)}")

        # Sort by predicted mask quality, highest first.
        filtered_masks = sorted(filtered_masks, key=lambda x: x['predicted_iou'], reverse=True)

        # Add masks one at a time without overwriting existing labels.
        for idx, mask_data in enumerate(filtered_masks, start=1):
            mask = mask_data['segmentation']
            # Assign new labels only to unlabeled pixels.
            instance_map[mask & (instance_map == 0)] = idx

        postprocess_time = time.time() - postprocess_start
        logger.info(f"Post-processing completed in {postprocess_time:.2f}s")

        # Count detected cells.
        num_cells = len(np.unique(instance_map)) - 1
        total_time = time.time() - start_time
        logger.info(f"CellSAM detected {num_cells} cells in {total_time:.2f}s")

        # Update the progress bar.
        if progress_bar is not None:
            progress_bar.progress(1.0)

        return instance_map

    except Exception as e:
        logger.error(f"CellSAM segmentation failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise RuntimeError(f"CellSAM segmentation error: {str(e)}")
