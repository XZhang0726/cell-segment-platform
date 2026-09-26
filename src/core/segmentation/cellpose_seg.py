"""
Cellpose cell segmentation integration.

Segment cells using pretrained Cellpose models.
"""
import os
# Allow duplicate OpenMP libraries to work around library conflicts.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
from loguru import logger
import multiprocessing

try:
    from cellpose import models
    CELLPOSE_AVAILABLE = True
except Exception as e:
    CELLPOSE_AVAILABLE = False
    logger.warning(f"Cellpose not available: {type(e).__name__}: {str(e)}")


class StreamlitProgressBar:
    """Adapt the QProgressBar interface for Cellpose progress callbacks."""
    def __init__(self, streamlit_progress_bar=None):
        self._value = 0
        self._maximum = 100
        self._streamlit_bar = streamlit_progress_bar

    def setValue(self, value):
        """Set the progress value."""
        self._value = value
        if self._streamlit_bar is not None:
            # Compute fractional progress and update the Streamlit progress bar.
            progress = min(1.0, max(0.0, value / self._maximum))
            self._streamlit_bar.progress(progress)

    def setMaximum(self, maximum):
        """Set the maximum progress value."""
        self._maximum = maximum

    def value(self):
        """Get the current progress value."""
        return self._value

    def maximum(self):
        """Get the maximum progress value."""
        return self._maximum


def cellpose_segment(image: np.ndarray,
                     model_type: str = 'cyto2',
                     diameter: float = None,
                     channels: list = None,
                     progress_bar=None,
                     use_gpu: bool = False,
                     batch_size: int = 8,
                     normalize: dict = None) -> np.ndarray:
    """
    Segment cells using Cellpose.

    Args:
        image: Input image (H, W) or (H, W, C).
        model_type: Model type ('cyto', 'cyto2', 'nuclei').
        diameter: Cell diameter in pixels; None requests automatic estimation.
        channels: Channel configuration [cytoplasm channel, nucleus channel].
                 Grayscale: [0, 0].
                 RGB: [2, 3] (green for cytoplasm, blue for nuclei).
        progress_bar: Streamlit progress bar object.
        use_gpu: Enable GPU acceleration when available.
        batch_size: Batch size for processing large images (default: 8).
        normalize: Normalization parameter dictionary, e.g. {"tile_norm_blocksize": 0}.
                  None uses default normalization.

    Returns:
        Segmentation mask with a unique label for each cell.
    """
    if not CELLPOSE_AVAILABLE:
        raise ImportError("Cellpose not installed")

    # Get the CPU core count for thread configuration.
    cpu_count = multiprocessing.cpu_count()
    logger.info(f"Cellpose segmentation: model={model_type}, diameter={diameter}, CPU cores={cpu_count}")

    # Default channel configuration.
    if channels is None:
        if len(image.shape) == 2:
            channels = [0, 0]  # Grayscale image.
        else:
            channels = [0, 0]  # Use grayscale by default for RGB images.

    # Load the model with the requested GPU setting.
    model = models.CellposeModel(gpu=use_gpu, model_type=model_type)

    # Create the progress callback adapter.
    progress_callback = None
    if progress_bar is not None:
        progress_callback = StreamlitProgressBar(progress_bar)

    # Run segmentation; Cellpose 4.x returns masks, flows, and styles.
    # Build evaluation arguments.
    eval_params = {
        'diameter': diameter,
        'channels': channels,
        'flow_threshold': 0.4,
        'cellprob_threshold': 0.0,
        'progress': progress_callback,
        'batch_size': batch_size
    }

    # Include normalization parameters when provided.
    if normalize is not None:
        eval_params['normalize'] = normalize

    masks, flows, styles = model.eval(image, **eval_params)

    logger.debug(f"Cellpose detected {len(np.unique(masks))-1} cells")

    return masks
