"""
Unified cell segmentation interface.

Combines classical image processing and deep learning methods.
"""
import numpy as np
from pathlib import Path
from typing import Optional, Union, Literal
from enum import Enum

from ..core.segmentation.threshold import ThresholdSegmentation
from ..core.segmentation.watershed import WatershedSegmentation
from ..core.segmentation.edge import EdgeDetection
from ..core.segmentation.cellpose_seg import cellpose_segment
from ..core.segmentation.cellvit_seg import cellvit_segment
from ..core.segmentation.cellsam_seg import cellsam_segment
from ..inference.predictor import Predictor
from ..core.utils.logger import get_logger

logger = get_logger(__name__)


class SegmentationMethod(Enum):
    """Supported segmentation methods."""
    OTSU = "otsu"
    ADAPTIVE = "adaptive"
    WATERSHED = "watershed"
    EDGE_CANNY = "edge_canny"
    DEEP_LEARNING = "deep_learning"
    CELLPOSE = "cellpose"
    CELLVIT = "cellvit"
    CELLSAM = "cellsam"


class CellSegmenter:
    """
    Unified cell segmenter.

    Provides a common interface for classical and deep learning methods.
    """

    def __init__(
        self,
        method: Union[str, SegmentationMethod] = SegmentationMethod.OTSU,
        model_path: Optional[str] = None,
        device: str = "cuda"
    ):
        """
        Initialize the segmenter.

        Args:
            method: Segmentation method ("otsu", "adaptive", "watershed", "edge_canny", "deep_learning").
            model_path: Pretrained model path, used only by deep_learning.
            device: Execution device, used only by deep_learning.
        """
        if isinstance(method, str):
            method = SegmentationMethod(method)

        self.method = method
        self.model_path = model_path
        self.device = device

        # Initialize classical segmentation handlers.
        self.threshold_seg = ThresholdSegmentation()
        self.watershed_seg = WatershedSegmentation()
        self.edge_detector = EdgeDetection()

        # Initialize the deep learning predictor when needed.
        self.predictor = None
        if self.method == SegmentationMethod.DEEP_LEARNING:
            if model_path is None:
                logger.warning("Deep learning method selected but no model_path provided. "
                             "Using randomly initialized weights.")
            self.predictor = Predictor(model_path=model_path, device=device)
            logger.info("Initialized deep learning predictor")

        logger.info(f"CellSegmenter initialized with method: {self.method.value}")

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert an image to grayscale."""
        import cv2
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def segment(
        self,
        image: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Segment an image.

        Args:
            image: Input image (H, W, C) or (H, W).
            **kwargs: Method-specific parameters.

        Returns:
            Segmentation mask (H, W).
        """
        if self.method == SegmentationMethod.OTSU:
            return self._segment_otsu(image, **kwargs)
        elif self.method == SegmentationMethod.ADAPTIVE:
            return self._segment_adaptive(image, **kwargs)
        elif self.method == SegmentationMethod.WATERSHED:
            return self._segment_watershed(image, **kwargs)
        elif self.method == SegmentationMethod.EDGE_CANNY:
            return self._segment_edge(image, **kwargs)
        elif self.method == SegmentationMethod.CELLPOSE:
            return self._segment_cellpose(image, **kwargs)
        elif self.method == SegmentationMethod.CELLVIT:
            return self._segment_cellvit(image, **kwargs)
        elif self.method == SegmentationMethod.CELLSAM:
            return self._segment_cellsam(image, **kwargs)
        elif self.method == SegmentationMethod.DEEP_LEARNING:
            return self._segment_deep_learning(image, **kwargs)
        else:
            raise ValueError(f"Unknown segmentation method: {self.method}")

    def _segment_otsu(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Segment using Otsu thresholding."""
        gray = self._to_grayscale(image)
        return self.threshold_seg.otsu_threshold(gray)

    def _segment_adaptive(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Segment using adaptive thresholding."""
        gray = self._to_grayscale(image)
        block_size = kwargs.get('block_size', 11)
        C = kwargs.get('C', 2)
        return self.threshold_seg.adaptive_threshold(gray, block_size=block_size, C=C)

    def _segment_watershed(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Segment using watershed segmentation."""
        # Convert to grayscale.
        gray = self._to_grayscale(image)
        # Binarize.
        binary = self.threshold_seg.otsu_threshold(gray)
        # Apply watershed segmentation.
        return self.watershed_seg.watershed_distance_transform(binary)

    def _segment_edge(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Segment using edge detection."""
        gray = self._to_grayscale(image)
        low_threshold = kwargs.get('low_threshold', 50)
        high_threshold = kwargs.get('high_threshold', 150)
        return self.edge_detector.canny(gray, threshold1=low_threshold, threshold2=high_threshold)

    def _segment_deep_learning(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Segment using a deep learning model."""
        if self.predictor is None:
            raise RuntimeError("Deep learning predictor not initialized. "
                             "Please provide model_path when creating CellSegmenter.")

        target_size = kwargs.get('target_size', (256, 256))
        threshold = kwargs.get('threshold', 0.5)
        return_original_size = kwargs.get('return_original_size', True)

        return self.predictor.predict(
            image,
            target_size=target_size,
            threshold=threshold,
            return_original_size=return_original_size
        )

    def _segment_cellpose(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Segment using a Cellpose model."""
        model_type = kwargs.get('model_type', 'cyto2')
        diameter = kwargs.get('diameter', None)
        channels = kwargs.get('channels', None)
        progress_bar = kwargs.get('progress_bar', None)
        use_gpu = kwargs.get('use_gpu', False)

        return cellpose_segment(
            image,
            model_type=model_type,
            diameter=diameter,
            channels=channels,
            progress_bar=progress_bar,
            use_gpu=use_gpu
        )

    def _segment_cellvit(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Segment using a CellViT model."""
        model_type = kwargs.get('model_type', 'CellViT-256')
        use_gpu = kwargs.get('use_gpu', False)
        target_size = kwargs.get('target_size', 256)
        progress_bar = kwargs.get('progress_bar', None)

        return cellvit_segment(
            image,
            model_type=model_type,
            use_gpu=use_gpu,
            target_size=target_size,
            progress_bar=progress_bar
        )

    def _segment_cellsam(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Generate masks using SAM through the legacy cellsam interface."""
        model_type = kwargs.get('model_type', 'vit_b')
        use_gpu = kwargs.get('use_gpu', False)
        points_per_side = kwargs.get('points_per_side', 32)
        progress_bar = kwargs.get('progress_bar', None)

        return cellsam_segment(
            image,
            model_type=model_type,
            use_gpu=use_gpu,
            points_per_side=points_per_side,
            progress_bar=progress_bar
        )
