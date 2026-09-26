"""
Demonstration of classical segmentation algorithms

Demonstrate thresholding, edge detection, morphology, watershed segmentation, and contour analysis
"""
import numpy as np
import cv2
from pathlib import Path

from src.data.image_io import ImageIO
from src.core.segmentation.threshold import ThresholdSegmentation
from src.core.segmentation.edge import EdgeDetection
from src.core.segmentation.morphology import MorphologicalOps
from src.core.segmentation.watershed import WatershedSegmentation
from src.core.segmentation.contour import ContourAnalysis
from src.core.utils.logger import setup_logger
from src.core.utils.paths import ensure_dir

# Set up logging
logger = setup_logger(level="INFO")
logger.info("Starting the classical segmentation demonstration...")

# Create the test output directory
test_dir = ensure_dir("test_segmentation_output")
logger.info(f"Test output directory: {test_dir}")

# 1. Create a test image
logger.info("\n=== Step 1: Create a test image ===")
height, width = 256, 256
test_image = np.zeros((height, width, 3), dtype=np.uint8)

# Add geometric shapes to simulate cells
cv2.circle(test_image, (64, 64), 30, (200, 200, 200), -1)
cv2.circle(test_image, (192, 64), 25, (200, 200, 200), -1)
cv2.circle(test_image, (64, 192), 28, (200, 200, 200), -1)
cv2.circle(test_image, (192, 192), 32, (200, 200, 200), -1)
cv2.circle(test_image, (128, 128), 35, (200, 200, 200), -1)

# Add touching cells
cv2.ellipse(test_image, (100, 180), (25, 15), 45, 0, 360, (200, 200, 200), -1)
cv2.ellipse(test_image, (120, 190), (20, 12), -30, 0, 360, (200, 200, 200), -1)

# Add a small amount of noise
noise = np.random.randint(0, 30, test_image.shape, dtype=np.uint8)
test_image = cv2.add(test_image, noise)

logger.info(f"Created test image: shape={test_image.shape}")

# Save the original image
ImageIO.save_image(test_image, test_dir / "01_original.png")
logger.info("[OK] Saved the original image")

# Convert to grayscale
gray_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
ImageIO.save_image(gray_image, test_dir / "02_grayscale.png")
logger.info("[OK] Saved the grayscale image")

# 2. Threshold segmentation demonstration
logger.info("\n=== Step 2: Threshold segmentation demonstration ===")

# Otsu threshold segmentation
otsu_binary = ThresholdSegmentation.otsu_threshold(gray_image)
ImageIO.save_image(otsu_binary, test_dir / "03_threshold_otsu.png")
logger.info("[OK] Otsu threshold segmentation completed")

# Fixed-threshold segmentation
fixed_binary = ThresholdSegmentation.fixed_threshold(gray_image, threshold=100)
ImageIO.save_image(fixed_binary, test_dir / "04_threshold_fixed.png")
logger.info("[OK] Fixed-threshold segmentation completed")

# Adaptive threshold segmentation
adaptive_binary = ThresholdSegmentation.adaptive_threshold(gray_image, block_size=15)
ImageIO.save_image(adaptive_binary, test_dir / "05_threshold_adaptive.png")
logger.info("[OK] Adaptive threshold segmentation completed")

# 3. Edge detection demonstration
logger.info("\n=== Step 3: Edge detection demonstration ===")

# Canny edge detection
canny_edges = EdgeDetection.canny(gray_image, threshold1=50, threshold2=150)
ImageIO.save_image(canny_edges, test_dir / "06_edge_canny.png")
logger.info("[OK] Canny edge detection completed")

# Sobel edge detection
sobel_edges = EdgeDetection.sobel(gray_image, dx=1, dy=1, ksize=3)
ImageIO.save_image(sobel_edges, test_dir / "07_edge_sobel.png")
logger.info("[OK] Sobel edge detection completed")

# Laplacian edge detection
laplacian_edges = EdgeDetection.laplacian(gray_image, ksize=3)
ImageIO.save_image(laplacian_edges, test_dir / "08_edge_laplacian.png")
logger.info("[OK] Laplacian edge detection completed")

# 4. Morphological operations demonstration
logger.info("\n=== Step 4: Morphological operations demonstration ===")

# Apply morphological operations to the Otsu binary mask
# Erosion
eroded = MorphologicalOps.erode(otsu_binary, kernel_size=(5, 5))
ImageIO.save_image(eroded, test_dir / "09_morph_erode.png")
logger.info("[OK] Erosion completed")

# Dilation
dilated = MorphologicalOps.dilate(otsu_binary, kernel_size=(5, 5))
ImageIO.save_image(dilated, test_dir / "10_morph_dilate.png")
logger.info("[OK] Dilation completed")

# Morphological opening
opened = MorphologicalOps.opening(otsu_binary, kernel_size=(5, 5))
ImageIO.save_image(opened, test_dir / "11_morph_opening.png")
logger.info("[OK] Morphological opening completed")

# Morphological closing
closed = MorphologicalOps.closing(otsu_binary, kernel_size=(5, 5))
ImageIO.save_image(closed, test_dir / "12_morph_closing.png")
logger.info("[OK] Morphological closing completed")

# 5. Watershed segmentation demonstration
logger.info("\n=== Step 5: Watershed segmentation demonstration ===")

# Watershed segmentation using a distance transform
watershed_labels = WatershedSegmentation.watershed_distance_transform(
    closed, min_distance=20
)
# Visualize the watershed result
watershed_vis = WatershedSegmentation.visualize_watershed(
    test_image, watershed_labels, show_boundaries=True
)
ImageIO.save_image(watershed_vis, test_dir / "13_watershed_distance.png")
logger.info("[OK] Distance-transform watershed segmentation completed")

# Marker-controlled watershed segmentation
watershed_markers = WatershedSegmentation.watershed_marker_controlled(
    gray_image, closed, sure_fg_erosion=5, sure_bg_dilation=5
)
watershed_vis2 = WatershedSegmentation.visualize_watershed(
    test_image, watershed_markers, show_boundaries=True
)
ImageIO.save_image(watershed_vis2, test_dir / "14_watershed_marker.png")
logger.info("[OK] Marker-controlled watershed segmentation completed")

# 6. Contour detection and analysis demonstration
logger.info("\n=== Step 6: Contour detection and analysis demonstration ===")

# Find contours
contours = ContourAnalysis.find_contours(closed, mode='external')
logger.info(f"Found {len(contours)} contours")

# Filter small contours
filtered_contours = ContourAnalysis.filter_contours(contours, min_area=200)
logger.info(f"Contours remaining after filtering: {len(filtered_contours)}")

# Draw contours
contour_image = ContourAnalysis.draw_contours(
    test_image, filtered_contours, color=(0, 255, 0), thickness=2
)
ImageIO.save_image(contour_image, test_dir / "15_contours.png")
logger.info("[OK] Contour drawing completed")

# Analyze contour properties
if len(filtered_contours) > 0:
    logger.info("\nContour property analysis:")
    for i, contour in enumerate(filtered_contours[:5]):  # Show only the first five contours
        props = ContourAnalysis.get_contour_properties(contour)
        logger.info(f"  Contour {i+1}:")
        logger.info(f"    - Area: {props['area']:.2f}")
        logger.info(f"    - Perimeter: {props['perimeter']:.2f}")
        logger.info(f"    - Circularity: {props['circularity']:.3f}")
        logger.info(f"    - Aspect ratio: {props['aspect_ratio']:.3f}")

# Get bounding boxes
boxes = ContourAnalysis.get_bounding_boxes(filtered_contours)
bbox_image = test_image.copy()
for box in boxes:
    x, y, w, h = box
    cv2.rectangle(bbox_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
ImageIO.save_image(bbox_image, test_dir / "16_bounding_boxes.png")
logger.info("[OK] Bounding box drawing completed")

# Summary
logger.info("\n" + "="*50)
logger.info("[OK] All classical segmentation demonstrations completed.")
logger.info(f"Test results saved to: {test_dir.absolute()}")
logger.info("="*50)
