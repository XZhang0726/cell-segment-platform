"""
Run a Cellpose smoke test and report the available CUDA environment.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
from cellpose import models
import time

print("=" * 60)
print("CELLPOSE AND CUDA ENVIRONMENT SMOKE TEST")
print("=" * 60)

# Test 1: PyTorch GPU Detection
print("\n[1] PyTorch GPU Detection:")
print(f"    PyTorch version: {torch.__version__}")
print(f"    CUDA available: {torch.cuda.is_available()}")
print(f"    CUDA version: {torch.version.cuda}")
print(f"    GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"    GPU name: {torch.cuda.get_device_name(0)}")
    print(f"    GPU compute capability: {torch.cuda.get_device_capability(0)}")
    print(f"    [OK] A CUDA device was detected.")

# Test 2: Cellpose Model Initialization
print("\n[2] Cellpose Model Initialization:")
try:
    model = models.CellposeModel(gpu=True)
    print(f"    [OK] Cellpose initialized with GPU acceleration requested.")
except Exception as e:
    print(f"    [ERROR] Error: {e}")
    exit(1)

# Test 3: Segmentation Smoke Test
print("\n[3] Segmentation Smoke Test:")
print("    Creating test image (512x512)...")
test_image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)

print("    Running segmentation with GPU acceleration requested...")
start_time = time.time()
try:
    masks, flows, styles = model.eval(test_image, diameter=30, channels=[0, 0])
    gpu_time = time.time() - start_time
    print(f"    [OK] Segmentation completed in {gpu_time:.3f} seconds")
    print(f"    [OK] Output shape: {masks.shape}")
except Exception as e:
    print(f"    [ERROR] Error during segmentation: {e}")
    exit(1)

# Final Summary
print("\n" + "=" * 60)
print("FINAL RESULT: SEGMENTATION SMOKE TEST COMPLETED")
print("=" * 60)
print("[OK] Cellpose completed inference on the synthetic test image.")
print("[INFO] PyTorch and CUDA versions are reported above.")
print("[INFO] A GPU request alone does not establish which device performed inference.")
print("=" * 60)
print("\nInspect the detected device and backend logs before interpreting GPU performance.")
print("Run this script in the environment containing your Cellpose installation.")
