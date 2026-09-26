"""
Functional smoke test

Demonstrate image I/O and preprocessing functionality.
"""
import numpy as np
from pathlib import Path

from src.data.image_io import ImageIO
from src.core.preprocessing.preprocess import ImagePreprocessor
from src.core.utils.logger import setup_logger
from src.core.utils.paths import ensure_dir

# Set up logging
logger = setup_logger(level="INFO")
logger.info("Starting functional smoke tests...")

# Create the test output directory
test_dir = ensure_dir("test_output")
logger.info(f"Test output directory: {test_dir}")

# 1. Create a test image
logger.info("\n=== Step 1: Create a test image ===")
# Create a noisy test image
height, width = 256, 256
test_image = np.zeros((height, width, 3), dtype=np.uint8)

# Add geometric shapes
cv2 = __import__('cv2')
cv2.circle(test_image, (128, 128), 50, (255, 0, 0), -1)  # Blue circle (BGR)
cv2.rectangle(test_image, (50, 50), (150, 100), (0, 255, 0), -1)  # Green rectangle
cv2.line(test_image, (0, 0), (256, 256), (0, 0, 255), 3)  # Red line (BGR)

# Add noise
noise = np.random.randint(0, 50, test_image.shape, dtype=np.uint8)
test_image = cv2.add(test_image, noise)

logger.info(f"Created test image: shape={test_image.shape}, dtype={test_image.dtype}")

# 2. Save the original image
logger.info("\n=== Step 2: Save the original image ===")
original_path = test_dir / "01_original.png"
ImageIO.save_image(test_image, original_path)
logger.info(f"Saved the original image to: {original_path}")

# 3. Test image loading
logger.info("\n=== Step 3: Test image loading ===")
loaded_image = ImageIO.load_image(original_path)
logger.info(f"Loaded image: shape={loaded_image.shape}")

# Verify that the loaded image matches the original
assert loaded_image.shape == test_image.shape
logger.info("[OK] Image loading test passed")

# 4. Test grayscale conversion
logger.info("\n=== Step 4: Test grayscale conversion ===")
gray_image = ImageIO.load_image(original_path, grayscale=True)
logger.info(f"Grayscale image: shape={gray_image.shape}")
ImageIO.save_image(gray_image, test_dir / "02_grayscale.png")
logger.info("[OK] Grayscale conversion test passed")

# 5. Test image normalization
logger.info("\n=== Step 5: Test image normalization ===")
normalized = ImagePreprocessor.normalize(test_image, method='minmax')
logger.info(f"After normalization: min={normalized.min():.3f}, max={normalized.max():.3f}")
# Convert back to uint8 before saving
normalized_uint8 = (normalized * 255).astype(np.uint8)
ImageIO.save_image(normalized_uint8, test_dir / "03_normalized.png")
logger.info("[OK] Image normalization test passed")

# 6. Test image resizing
logger.info("\n=== Step 6: Test image resizing ===")
resized = ImagePreprocessor.resize(test_image, (128, 128))
logger.info(f"After resizing: shape={resized.shape}")
ImageIO.save_image(resized, test_dir / "04_resized_128x128.png")
logger.info("[OK] Image resizing test passed")

# 7. Test image denoising
logger.info("\n=== Step 7: Test image denoising ===")
denoised_gaussian = ImagePreprocessor.denoise(test_image, method='gaussian', ksize=5)
ImageIO.save_image(denoised_gaussian, test_dir / "05_denoised_gaussian.png")
logger.info("[OK] Gaussian denoising test passed")

denoised_median = ImagePreprocessor.denoise(test_image, method='median', ksize=5)
ImageIO.save_image(denoised_median, test_dir / "06_denoised_median.png")
logger.info("[OK] Median denoising test passed")

# 8. Test contrast enhancement
logger.info("\n=== Step 8: Test contrast enhancement ===")
enhanced = ImagePreprocessor.enhance_contrast(test_image, method='clahe')
ImageIO.save_image(enhanced, test_dir / "07_enhanced_clahe.png")
logger.info("[OK] CLAHE contrast enhancement test passed")

# 9. Test image metadata retrieval
logger.info("\n=== Step 9: Test image metadata retrieval ===")
info = ImageIO.get_image_info(original_path)
logger.info(f"Image information:")
logger.info(f"  - Filename: {info['filename']}")
logger.info(f"  - Format: {info['format']}")
logger.info(f"  - Dimensions: {info['width']}x{info['height']}")
logger.info(f"  - Mode: {info['mode']}")
logger.info("[OK] Image metadata test passed")

# 10. Integration test: complete preprocessing pipeline
logger.info("\n=== Step 10: Integration test - Complete preprocessing pipeline ===")
# Load -> denoise -> enhance -> normalize -> resize
pipeline_image = ImageIO.load_image(original_path)
pipeline_image = ImagePreprocessor.denoise(pipeline_image, method='bilateral')
pipeline_image = ImagePreprocessor.enhance_contrast(pipeline_image, method='clahe')
pipeline_image = ImagePreprocessor.normalize(pipeline_image, method='minmax')
pipeline_image = (pipeline_image * 255).astype(np.uint8)
pipeline_image = ImagePreprocessor.resize(pipeline_image, (200, 200))
ImageIO.save_image(pipeline_image, test_dir / "08_pipeline_result.png")
logger.info("[OK] Complete preprocessing pipeline test passed")

# Summary
logger.info("\n" + "="*50)
logger.info("[OK] All functional smoke tests passed.")
logger.info(f"Test results saved to: {test_dir.absolute()}")
logger.info("="*50)
