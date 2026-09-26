"""
Unit tests for classical segmentation algorithms
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.core.segmentation.threshold import ThresholdSegmentation
from src.core.segmentation.edge import EdgeDetection
from src.core.segmentation.morphology import MorphologicalOps
from src.core.segmentation.watershed import WatershedSegmentation
from src.core.segmentation.contour import ContourAnalysis


class TestThresholdSegmentation:
    """Test threshold segmentation"""

    @pytest.fixture
    def sample_gray_image(self):
        """Create a grayscale test image"""
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        return image

    def test_otsu_threshold(self, sample_gray_image):
        """Test Otsu threshold segmentation"""
        binary = ThresholdSegmentation.otsu_threshold(sample_gray_image)

        # Verify that the output image is binary
        assert binary.shape == sample_gray_image.shape
        assert set(np.unique(binary)).issubset({0, 255})

    def test_otsu_threshold_return_value(self, sample_gray_image):
        """Test that Otsu segmentation returns the threshold"""
        binary, threshold = ThresholdSegmentation.otsu_threshold(
            sample_gray_image, return_threshold=True
        )

        # Verify that the threshold is returned
        assert isinstance(threshold, (int, float))
        assert 0 <= threshold <= 255

    def test_fixed_threshold(self, sample_gray_image):
        """Test fixed-threshold segmentation"""
        binary = ThresholdSegmentation.fixed_threshold(
            sample_gray_image, threshold=127
        )

        # Verify that the output image is binary
        assert binary.shape == sample_gray_image.shape
        assert set(np.unique(binary)).issubset({0, 255})

    def test_adaptive_threshold(self, sample_gray_image):
        """Test adaptive thresholding"""
        binary = ThresholdSegmentation.adaptive_threshold(
            sample_gray_image, block_size=11
        )

        # Verify that the output image is binary
        assert binary.shape == sample_gray_image.shape
        assert set(np.unique(binary)).issubset({0, 255})

    def test_threshold_invalid_input(self):
        """Test invalid input"""
        # Color input should raise an error
        color_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        with pytest.raises(ValueError):
            ThresholdSegmentation.otsu_threshold(color_image)


class TestEdgeDetection:
    """Test edge detection algorithms"""

    @pytest.fixture
    def sample_gray_image(self):
        """Create a grayscale test image"""
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        return image

    def test_canny_edge(self, sample_gray_image):
        """Test Canny edge detection"""
        edges = EdgeDetection.canny(sample_gray_image, threshold1=50, threshold2=150)

        # Verify the output shape
        assert edges.shape == sample_gray_image.shape
        # Verify that the image is binary
        assert set(np.unique(edges)).issubset({0, 255})

    def test_sobel_edge(self, sample_gray_image):
        """Test Sobel edge detection"""
        edges = EdgeDetection.sobel(sample_gray_image, dx=1, dy=1, ksize=3)

        # Verify the output shape
        assert edges.shape == sample_gray_image.shape
        assert edges.dtype == np.uint8

    def test_laplacian_edge(self, sample_gray_image):
        """Test Laplacian edge detection"""
        edges = EdgeDetection.laplacian(sample_gray_image, ksize=3)

        # Verify the output shape
        assert edges.shape == sample_gray_image.shape
        assert edges.dtype == np.uint8

    def test_scharr_edge(self, sample_gray_image):
        """Test Scharr edge detection"""
        edges = EdgeDetection.scharr(sample_gray_image, dx=1, dy=0)

        # Verify the output shape
        assert edges.shape == sample_gray_image.shape
        assert edges.dtype == np.uint8

    def test_edge_invalid_input(self):
        """Test invalid input"""
        # Color input should raise an error
        color_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        with pytest.raises(ValueError):
            EdgeDetection.canny(color_image)


class TestMorphologicalOps:
    """Test morphological operations"""

    @pytest.fixture
    def sample_binary_image(self):
        """Create a binary test image"""
        image = np.zeros((100, 100), dtype=np.uint8)
        image[30:70, 30:70] = 255
        return image

    def test_erode(self, sample_binary_image):
        """Test erosion"""
        eroded = MorphologicalOps.erode(sample_binary_image, kernel_size=(5, 5))

        # Verify the output shape
        assert eroded.shape == sample_binary_image.shape
        # Erosion should reduce the white foreground area
        assert np.sum(eroded) < np.sum(sample_binary_image)

    def test_dilate(self, sample_binary_image):
        """Test dilation"""
        dilated = MorphologicalOps.dilate(sample_binary_image, kernel_size=(5, 5))

        # Verify the output shape
        assert dilated.shape == sample_binary_image.shape
        # Dilation should increase the white foreground area
        assert np.sum(dilated) > np.sum(sample_binary_image)

    def test_opening(self, sample_binary_image):
        """Test morphological opening"""
        opened = MorphologicalOps.opening(sample_binary_image, kernel_size=(5, 5))

        # Verify the output shape
        assert opened.shape == sample_binary_image.shape

    def test_closing(self, sample_binary_image):
        """Test morphological closing"""
        closed = MorphologicalOps.closing(sample_binary_image, kernel_size=(5, 5))

        # Verify the output shape
        assert closed.shape == sample_binary_image.shape

    def test_gradient(self, sample_binary_image):
        """Test the morphological gradient"""
        gradient = MorphologicalOps.gradient(sample_binary_image, kernel_size=(5, 5))

        # Verify the output shape
        assert gradient.shape == sample_binary_image.shape


class TestWatershedSegmentation:
    """Test watershed segmentation"""

    @pytest.fixture
    def sample_binary_image(self):
        """Create a binary test image"""
        image = np.zeros((100, 100), dtype=np.uint8)
        # Create two separate circles
        import cv2
        cv2.circle(image, (30, 30), 15, 255, -1)
        cv2.circle(image, (70, 70), 15, 255, -1)
        return image

    @pytest.fixture
    def sample_color_image(self):
        """Create a color test image"""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        return image

    def test_watershed_distance_transform(self, sample_binary_image):
        """Test distance-transform watershed segmentation"""
        labels = WatershedSegmentation.watershed_distance_transform(
            sample_binary_image, min_distance=10
        )

        # Verify the output shape
        assert labels.shape == sample_binary_image.shape
        # There should be multiple regions
        assert len(np.unique(labels)) > 1

    def test_watershed_marker_controlled(self, sample_color_image, sample_binary_image):
        """Test marker-controlled watershed segmentation"""
        markers = WatershedSegmentation.watershed_marker_controlled(
            sample_color_image, sample_binary_image
        )

        # Verify the output shape
        assert markers.shape == sample_color_image.shape[:2]


class TestContourAnalysis:
    """Test contour detection and analysis"""

    @pytest.fixture
    def sample_binary_image(self):
        """Create a binary test image"""
        image = np.zeros((100, 100), dtype=np.uint8)
        # Create several rectangles
        import cv2
        cv2.rectangle(image, (10, 10), (30, 30), 255, -1)
        cv2.rectangle(image, (50, 50), (80, 80), 255, -1)
        return image

    def test_find_contours(self, sample_binary_image):
        """Test contour detection"""
        contours = ContourAnalysis.find_contours(sample_binary_image)

        # At least one contour should be found
        assert len(contours) > 0
        # Each contour should be a NumPy array
        assert all(isinstance(c, np.ndarray) for c in contours)

    def test_filter_contours(self, sample_binary_image):
        """Test contour filtering"""
        contours = ContourAnalysis.find_contours(sample_binary_image)
        filtered = ContourAnalysis.filter_contours(contours, min_area=100)

        # Filtering should not increase the number of contours
        assert len(filtered) <= len(contours)

    def test_get_contour_properties(self, sample_binary_image):
        """Test contour property extraction"""
        contours = ContourAnalysis.find_contours(sample_binary_image)
        if len(contours) > 0:
            properties = ContourAnalysis.get_contour_properties(contours[0])

            # Verify the returned properties
            assert 'area' in properties
            assert 'perimeter' in properties
            assert 'circularity' in properties
            assert 'aspect_ratio' in properties
            assert properties['area'] >= 0

    def test_draw_contours(self, sample_binary_image):
        """Test contour drawing"""
        contours = ContourAnalysis.find_contours(sample_binary_image)
        result = ContourAnalysis.draw_contours(
            sample_binary_image, contours, color=(0, 255, 0)
        )

        # Verify that the output image is in color
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_get_bounding_boxes(self, sample_binary_image):
        """Test bounding box extraction"""
        contours = ContourAnalysis.find_contours(sample_binary_image)
        boxes = ContourAnalysis.get_bounding_boxes(contours)

        # The number of bounding boxes should equal the number of contours
        assert len(boxes) == len(contours)

