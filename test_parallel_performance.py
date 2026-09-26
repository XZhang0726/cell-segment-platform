"""
Compare sequential and parallel segmentation times on synthetic images.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
import numpy as np
import time
from PIL import Image

# Add the project directory to the import path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.api.segmentation import CellSegmenter, SegmentationMethod

def create_test_images(num_images=5, size=(512, 512)):
    """Create synthetic test images containing circular objects."""
    print(f"Creating {num_images} test images...")
    images = []
    for i in range(num_images):
        # Create an image containing random noise
        img = np.random.randint(0, 255, size, dtype=np.uint8)
        # Add circles to represent cells
        for _ in range(20):
            center_x = np.random.randint(50, size[0]-50)
            center_y = np.random.randint(50, size[1]-50)
            radius = np.random.randint(10, 30)
            y, x = np.ogrid[:size[0], :size[1]]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            img[mask] = 200
        images.append(img)
    return images

def test_sequential(images, method=SegmentationMethod.OTSU):
    """Test sequential processing"""
    print(f"\n{'='*50}")
    print(f"Sequential processing test - {len(images)} images")
    print(f"{'='*50}")

    segmenter = CellSegmenter(method=method)
    start_time = time.time()

    results = []
    for i, img in enumerate(images):
        print(f"Processing image {i+1}/{len(images)}...", end='\r')
        mask = segmenter.segment(img)
        results.append(mask)

    elapsed = time.time() - start_time
    print(f"\nSequential processing completed: {elapsed:.2f} seconds")
    print(f"Average time per image: {elapsed/len(images):.2f} seconds")

    return elapsed, results

def test_parallel(images, method=SegmentationMethod.OTSU):
    """Test parallel processing with a simulated batch"""
    print(f"\n{'='*50}")
    print(f"Parallel processing test - {len(images)} images")
    print(f"{'='*50}")

    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    cpu_count = multiprocessing.cpu_count()
    max_workers = max(1, cpu_count // 2)
    print(f"Using {max_workers} worker processes ({cpu_count} CPU cores available)")

    def process_image(img):
        segmenter = CellSegmenter(method=method)
        return segmenter.segment(img)

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_image, images))

    elapsed = time.time() - start_time
    print(f"Parallel processing completed: {elapsed:.2f} seconds")
    print(f"Average time per image: {elapsed/len(images):.2f} seconds")

    return elapsed, results

def main():
    print("="*60)
    print("Parallel cell segmentation benchmark")
    print("="*60)

    # Create the test images
    num_images = 8  # Number of test images
    images = create_test_images(num_images)

    # Use Otsu thresholding for a short benchmark
    method = SegmentationMethod.OTSU
    print(f"\nSegmentation method: Otsu thresholding")

    # Sequential benchmark
    time_sequential, _ = test_sequential(images, method)

    # Parallel benchmark
    time_parallel, _ = test_parallel(images, method)

    # Performance comparison
    print(f"\n{'='*60}")
    print("Performance comparison results")
    print(f"{'='*60}")
    print(f"Sequential processing: {time_sequential:.2f} seconds")
    print(f"Parallel processing: {time_parallel:.2f} seconds")

    if time_parallel < time_sequential:
        speedup = time_sequential / time_parallel
        improvement = ((time_sequential - time_parallel) / time_sequential) * 100
        print(f"\n[OK] Speedup: {speedup:.2f}x")
        print(f"[OK] Performance improvement: {improvement:.1f}%")
        print(f"[OK] Time saved: {time_sequential - time_parallel:.2f} seconds")
    else:
        print(f"\n[WARN] No parallel speedup observed (the batch or workload may be too small)")

    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()
