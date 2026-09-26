"""
Instance matching performance comparison

Compare the original and optimized implementations
"""
import time
import numpy as np
from pathlib import Path
import cv2
from loguru import logger

# Import the original and optimized implementations
from src.core.fusion.instance_matcher import match_instances as match_instances_original
from src.core.fusion.instance_matcher_optimized import match_instances_optimized


def load_test_image(image_path: str) -> np.ndarray:
    """Load a test image (supports Unicode paths)"""
    # Read the file with numpy.fromfile to support Unicode paths
    img_array = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Unable to load the image: {image_path}")

    # Convert to grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    logger.info(f"Loaded image: {image_path}, dimensions: {img.shape}")
    return img


def simulate_segmentation_masks(image: np.ndarray, num_models: int = 3) -> list:
    """
    Simulate segmentation predictions from multiple models

    Generate label maps for the performance benchmark.
    Each simulated model produces slightly different cell detections.
    """
    height, width = image.shape
    masks = []

    # Use threshold segmentation as the baseline
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Distance transform and watershed
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    for model_idx in range(num_models):
        # Use a slightly different threshold for each model
        threshold_ratio = 0.3 + model_idx * 0.1
        _, markers = cv2.threshold(dist_transform, threshold_ratio * dist_transform.max(), 255, 0)
        markers = markers.astype(np.uint8)

        # Connected-component labeling
        num_labels, labels = cv2.connectedComponents(markers)

        # Apply watershed segmentation
        labels = labels.astype(np.int32)
        cv2.watershed(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), labels)

        # Replace watershed boundary labels (-1) with background (0)
        labels[labels == -1] = 0

        masks.append(labels)
        logger.info(f"Model {model_idx}: detected {num_labels-1} instances")

    return masks


def run_performance_test(masks_list: list, iou_threshold: float = 0.5):
    """
    Run the performance comparison

    Args:
        masks_list: List of label maps from different segmentation models
        iou_threshold: IoU threshold for matching instances
    """
    logger.info("=" * 80)
    logger.info("Starting the performance comparison")
    logger.info("=" * 80)

    # Test the original implementation
    logger.info("\nOriginal implementation")
    start_time = time.time()
    try:
        results_original = match_instances_original(masks_list, iou_threshold)
        time_original = time.time() - start_time
        logger.info(f"[OK] Original implementation completed in {time_original:.4f} seconds")
        logger.info(f"  Matched {len(results_original)} instance groups")
    except Exception as e:
        logger.error(f"[FAIL] Original implementation failed: {e}")
        time_original = None
        results_original = None

    # Test the optimized implementation
    logger.info("\nOptimized implementation")
    start_time = time.time()
    try:
        results_optimized = match_instances_optimized(masks_list, iou_threshold)
        time_optimized = time.time() - start_time
        logger.info(f"[OK] Optimized implementation completed in {time_optimized:.4f} seconds")
        logger.info(f"  Matched {len(results_optimized)} instance groups")
    except Exception as e:
        logger.error(f"[FAIL] Optimized implementation failed: {e}")
        time_optimized = None
        results_optimized = None

    # Performance comparison
    logger.info("\n" + "=" * 80)
    logger.info("Performance comparison")
    logger.info("=" * 80)

    if time_original and time_optimized:
        speedup = time_original / time_optimized
        logger.info(f"Original implementation runtime: {time_original:.4f} seconds")
        logger.info(f"Optimized implementation runtime: {time_optimized:.4f} seconds")
        logger.info(f"Speedup: {speedup:.2f}x")
        logger.info(f"Performance improvement: {(speedup-1)*100:.1f}%")

        if speedup > 10:
            logger.info("[INFO] Substantial improvement")
        elif speedup > 5:
            logger.info("[OK] Good improvement")
        else:
            logger.info("[WARN] Limited improvement")

    # Result consistency check
    if results_original and results_optimized:
        logger.info("\nResult consistency check")
        if len(results_original) == len(results_optimized):
            logger.info(f"[OK] The numbers of matched groups agree: {len(results_original)}")
        else:
            logger.warning(f"[WARN] The numbers of matched groups differ: original={len(results_original)}, optimized={len(results_optimized)}")

    logger.info("=" * 80)


def main():
    """Main entry point"""
    # Find test images
    test_image_dir = Path(r"C:\Users\XB001\Desktop\cells")

    if not test_image_dir.exists():
        logger.error(f"The test image directory does not exist: {test_image_dir}")
        return

    # Find the first image
    image_files = list(test_image_dir.glob("*.jpg")) + \
                  list(test_image_dir.glob("*.png")) + \
                  list(test_image_dir.glob("*.tif"))

    if not image_files:
        logger.error(f"No image files found in {test_image_dir}")
        return

    test_image_path = str(image_files[0])
    logger.info(f"Using test image: {test_image_path}")

    # Load the image
    image = load_test_image(test_image_path)

    # Simulate segmentation predictions from multiple models
    logger.info("\nGenerating simulated segmentation results...")
    masks_list = simulate_segmentation_masks(image, num_models=3)

    # Run the performance benchmark
    run_performance_test(masks_list, iou_threshold=0.5)


if __name__ == "__main__":
    main()
