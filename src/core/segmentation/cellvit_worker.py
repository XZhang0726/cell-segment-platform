"""
CellViT worker script for the dedicated CellViT environment.

The main application invokes this script through a subprocess.
It loads image data, runs CellViT inference, and writes the result.
"""
import sys
import os
import pickle
import numpy as np
from pathlib import Path

# Allow duplicate OpenMP libraries to work around library conflicts.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def run_cellvit_inference(input_file, output_file):
    """
    Run CellViT inference.

    Args:
        input_file: Input pickle file containing the image and inference parameters.
        output_file: Output pickle file for inference results.
    """
    import time
    start_time = time.time()

    try:
        import torch
        import torch.nn.functional as F
        from cellvit.models.cell_segmentation.cellvit_256 import CellViT256
        from cellvit.utils.cache_models import cache_cellvit_256
        from cellvit.inference.postprocessing_numpy import DetectionCellPostProcessor
        import cv2

        # Read input data.
        with open(input_file, 'rb') as f:
            data = pickle.load(f)

        image = data['image']
        model_type = data['model_type']
        use_gpu = data['use_gpu']
        target_size = data['target_size']

        # Select the device.
        device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        # Print device information.
        print(f"[CellViT Worker] Device: {device}")
        print(f"[CellViT Worker] use_gpu: {use_gpu}")
        print(f"[CellViT Worker] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[CellViT Worker] GPU: {torch.cuda.get_device_name(0)}")
        sys.stdout.flush()

        # Load the model.
        print(f"[CellViT Worker] Loading model...")
        sys.stdout.flush()
        model_load_start = time.time()

        model_path = cache_cellvit_256()
        checkpoint = torch.load(model_path, map_location=device)

        model = CellViT256(
            model256_path=model_path,
            num_nuclei_classes=6,
            num_tissue_classes=19,
        )

        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model = model.to(device)
        model.eval()

        model_load_time = time.time() - model_load_start
        print(f"[CellViT Worker] Model loaded in {model_load_time:.2f}s")

        # Preprocess the image.
        h, w = image.shape[:2]
        original_shape = (h, w)

        # Compute padding.
        pad_h = max(0, target_size - h)
        pad_w = max(0, target_size - w)

        # Resize images larger than target_size.
        if h > target_size or w > target_size:
            scale = target_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            h, w = new_h, new_w
            pad_h = target_size - h
            pad_w = target_size - w

        # Padding
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Convert to a tensor.
        image_tensor = torch.from_numpy(image).float()
        image_tensor = image_tensor.permute(2, 0, 1)
        image_tensor = image_tensor / 255.0

        image_tensor = F.pad(
            image_tensor,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode='constant',
            value=0
        )

        image_tensor = image_tensor.unsqueeze(0).to(device)

        # Run inference.
        print(f"[CellViT Worker] Starting inference...")
        inference_start = time.time()

        with torch.no_grad():
            predictions = model.forward(image_tensor, retrieve_tokens=False)

        inference_time = time.time() - inference_start
        print(f"[CellViT Worker] Inference completed in {inference_time:.2f}s")

        # Print prediction keys for debugging.
        print(f"[CellViT Worker] Predictions keys: {list(predictions.keys())}")
        sys.stdout.flush()

        # Use the CellViT postprocessor to convert HV maps to instance labels.
        print(f"[CellViT Worker] Postprocessing HV maps into instance labels...")
        postprocess_start = time.time()

        # Convert prediction tensors to the postprocessor's required (B, H, W, C) format.
        # Current format: (B, C, H, W); required format: (B, H, W, C).
        predictions_reshaped = {}
        for key in ['nuclei_binary_map', 'hv_map', 'nuclei_type_map']:
            if key in predictions:
                tensor = predictions[key]
                print(f"[CellViT Worker] {key} original shape: {tensor.shape}")
                # Convert (B, C, H, W) to (B, H, W, C).
                if tensor.dim() == 4:
                    tensor = tensor.permute(0, 2, 3, 1)
                predictions_reshaped[key] = tensor
                print(f"[CellViT Worker] {key} converted shape: {tensor.shape}")

        sys.stdout.flush()

        # Create a minimal WSI metadata object required by the postprocessor.
        class SimpleWSI:
            def __init__(self):
                self.metadata = {'pixel_size': 1.0}

        # Create the postprocessor.
        # num_nuclei_classes = 6 (the CellViT-256 default).
        post_processor = DetectionCellPostProcessor(
            wsi=SimpleWSI(),
            nr_types=6,
            binary=False
        )

        # Run postprocessing.
        try:
            instance_maps, cell_dicts = post_processor.post_process_batch(predictions_reshaped)
            print(f"[CellViT Worker] Postprocessing completed in {time.time() - postprocess_start:.2f}s")

            # Extract the instance map for the first image.
            instance_map = instance_maps[0].cpu().numpy() if isinstance(instance_maps, torch.Tensor) else instance_maps[0]

            print(f"[CellViT Worker] Instance map shape: {instance_map.shape}")
            print(f"[CellViT Worker] Instance map dtype: {instance_map.dtype}")
            print(f"[CellViT Worker] Instance map range: {instance_map.min()} - {instance_map.max()}")
            print(f"[CellViT Worker] Unique cells: {len(np.unique(instance_map)) - 1}")
            sys.stdout.flush()

        except Exception as e:
            print(f"[CellViT Worker] Postprocessing failed: {str(e)}")
            print(f"[CellViT Worker] Falling back to the binary map")
            # Fall back to the binary map if postprocessing fails.
            if 'nuclei_binary_map' in predictions:
                instance_map = predictions['nuclei_binary_map'][0, 1].cpu().numpy()
            else:
                raise ValueError("Unable to generate an instance map")
            sys.stdout.flush()

        # Remove padding.
        if pad_bottom > 0:
            instance_map = instance_map[pad_top:-pad_bottom, :]
        else:
            instance_map = instance_map[pad_top:, :]

        if pad_right > 0:
            instance_map = instance_map[:, pad_left:-pad_right]
        else:
            instance_map = instance_map[:, pad_left:]

        # Resize to the original image dimensions.
        if instance_map.shape != original_shape:
            instance_map = cv2.resize(
                instance_map.astype(np.float32),
                (original_shape[1], original_shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        # Save the result.
        result = {
            'success': True,
            'mask': instance_map.astype(np.int32),
            'error': None
        }

        with open(output_file, 'wb') as f:
            pickle.dump(result, f)

        total_time = time.time() - start_time
        print(f"[CellViT Worker] Total processing time: {total_time:.2f}s")
        print(f"[CellViT Worker] Detected {len(np.unique(instance_map)) - 1} nuclei")

        return 0

    except Exception as e:
        # Save the error message and full traceback.
        import traceback
        error_traceback = traceback.format_exc()

        # Write to stderr so the parent process can capture the error.
        print(f"[CellViT Worker] ERROR: {str(e)}", file=sys.stderr)
        print(f"[CellViT Worker] TRACEBACK:\n{error_traceback}", file=sys.stderr)
        sys.stderr.flush()

        result = {
            'success': False,
            'mask': None,
            'error': f"{str(e)}\n\nTraceback:\n{error_traceback}"
        }

        with open(output_file, 'wb') as f:
            pickle.dump(result, f)

        return 1


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python cellvit_worker.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    exit_code = run_cellvit_inference(input_file, output_file)
    sys.exit(exit_code)
