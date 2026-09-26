"""
CellViT cell segmentation integration.

Segment nuclei using a pretrained CellViT model.
CellViT uses a vision transformer and is designed for histopathology images.

Notes:
- CellViT runs in a dedicated environment (env_cellvit) through a subprocess.
- The main application can run separately and invokes the CellViT environment.
- This interface adapts a histopathology model to smaller input images.
"""
import os
import sys
import subprocess
import pickle
import tempfile
import numpy as np
from loguru import logger
from pathlib import Path
from typing import Optional, Tuple


def cellvit_segment(
    image: np.ndarray,
    model_type: str = "CellViT-256",
    use_gpu: bool = False,
    target_size: int = 256,
    progress_bar=None
) -> np.ndarray:
    """
    Segment nuclei using CellViT.

    Run inference in the CellViT environment through a subprocess to isolate dependencies.

    Args:
        image: Input RGB image (H, W, C) or grayscale image (H, W).
        model_type: Model type; currently supports "CellViT-256".
        use_gpu: Enable GPU acceleration when available.
        target_size: Target image size (256x256 for CellViT-256).
        progress_bar: Streamlit progress bar object.

    Returns:
        Integer output mask. Successful instance postprocessing assigns unique
        nucleus labels; the worker may otherwise return its binary-map fallback.

    Raises:
        RuntimeError: If inference fails.
    """
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

    logger.info(f"CellViT segmentation: image_shape={original_shape}, target_size={target_size}")

    # Locate the project root and environment.
    project_root = Path(__file__).parent.parent.parent.parent
    cellvit_env_path = project_root / "env_cellvit"
    worker_script = project_root / "src" / "core" / "segmentation" / "cellvit_worker.py"

    # Check that the environment and worker script exist.
    if not cellvit_env_path.exists():
        raise RuntimeError(
            f"CellViT environment not found at {cellvit_env_path}\n"
            "Please create the environment first."
        )

    if not worker_script.exists():
        raise RuntimeError(
            f"CellViT worker script not found at {worker_script}"
        )

    # Locate the Python executable.
    if sys.platform == "win32":
        python_exe = cellvit_env_path / "python.exe"
        if not python_exe.exists():
            python_exe = cellvit_env_path / "Scripts" / "python.exe"
    else:
        python_exe = cellvit_env_path / "bin" / "python"

    if not python_exe.exists():
        raise RuntimeError(
            f"Python executable not found in CellViT environment at {python_exe}"
        )

    logger.info(f"Using CellViT environment: {cellvit_env_path}")
    logger.info(f"Python executable: {python_exe}")

    try:
        # Update the progress bar.
        if progress_bar is not None:
            progress_bar.progress(0.2)

        # Create temporary files to exchange data with the worker.
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as input_file:
            input_path = input_file.name
            # Prepare input data.
            input_data = {
                'image': image,
                'model_type': model_type,
                'use_gpu': use_gpu,
                'target_size': target_size
            }
            pickle.dump(input_data, input_file)

        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as output_file:
            output_path = output_file.name

        # Update the progress bar.
        if progress_bar is not None:
            progress_bar.progress(0.4)

        logger.info("Calling CellViT worker in subprocess...")

        # Invoke the worker script.
        result = subprocess.run(
            [str(python_exe), str(worker_script), input_path, output_path],
            capture_output=True,
            text=True,
            timeout=300  # Five-minute timeout.
        )

        # Update the progress bar.
        if progress_bar is not None:
            progress_bar.progress(0.8)

        # Check the worker execution result.
        if result.returncode != 0:
            error_msg = f"CellViT worker failed with return code {result.returncode}\n"
            error_msg += f"STDOUT: {result.stdout}\n"
            error_msg += f"STDERR: {result.stderr}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Log successful CellViT worker output for debugging.
        if result.stdout:
            logger.info(f"CellViT worker stdout:\n{result.stdout}")
        if result.stderr:
            logger.info(f"CellViT worker stderr:\n{result.stderr}")

        # Read the output data.
        with open(output_path, 'rb') as f:
            output_data = pickle.load(f)

        # Remove temporary files.
        try:
            os.unlink(input_path)
            os.unlink(output_path)
        except:
            pass

        # Update the progress bar.
        if progress_bar is not None:
            progress_bar.progress(1.0)

        # Check the result.
        if not output_data['success']:
            raise RuntimeError(f"CellViT inference failed: {output_data['error']}")

        mask = output_data['mask']
        num_cells = len(np.unique(mask)) - 1
        logger.info(f"CellViT detected {num_cells} cells")

        return mask

    except subprocess.TimeoutExpired:
        logger.error("CellViT worker timeout")
        raise RuntimeError("CellViT inference timeout (>5 minutes)")
    except Exception as e:
        logger.error(f"CellViT segmentation failed: {str(e)}")
        raise RuntimeError(f"CellViT segmentation error: {str(e)}")
