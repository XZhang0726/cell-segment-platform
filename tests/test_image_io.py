"""
Unit tests for image I/O
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.data.image_io import ImageIO, load_image, save_image


class TestImageIO:
    """Test the ImageIO class"""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_image(self):
        """Create a sample test image"""
        # Create a simple RGB image
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        return image

    @pytest.fixture
    def sample_gray_image(self):
        """Create a grayscale test image"""
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        return image

    def test_save_and_load_image(self, temp_dir, sample_image):
        """Test saving and loading images"""
        # Save the image
        image_path = temp_dir / "test_image.png"
        save_image(sample_image, image_path)

        # Verify that the file exists
        assert image_path.exists()

        # Load the image
        loaded_image = load_image(image_path)

        # Verify that the shapes agree
        assert loaded_image.shape == sample_image.shape

    def test_load_grayscale(self, temp_dir, sample_image):
        """Test loading a grayscale image"""
        # Save the color image
        image_path = temp_dir / "test_color.png"
        save_image(sample_image, image_path)

        # Load in grayscale mode
        gray_image = load_image(image_path, grayscale=True)

        # Verify that the image is grayscale
        assert gray_image.ndim == 2

    def test_load_nonexistent_file(self):
        """Test loading a missing file"""
        with pytest.raises(FileNotFoundError):
            load_image("nonexistent_file.png")

    def test_load_unsupported_format(self, temp_dir):
        """Test loading an unsupported format"""
        unsupported_file = temp_dir / "test.xyz"
        unsupported_file.touch()

        with pytest.raises(ValueError):
            load_image(unsupported_file)

    def test_get_image_info(self, temp_dir, sample_image):
        """Test image metadata retrieval"""
        # Save the image
        image_path = temp_dir / "test_info.png"
        save_image(sample_image, image_path)

        # Get metadata
        info = ImageIO.get_image_info(image_path)

        # Validate metadata
        assert info['width'] == 100
        assert info['height'] == 100
        assert info['format'] == 'PNG'

    def test_load_images_batch(self, temp_dir, sample_image):
        """Test batch image loading"""
        # Create multiple test images
        image_paths = []
        for i in range(3):
            path = temp_dir / f"test_{i}.png"
            save_image(sample_image, path)
            image_paths.append(path)

        # Batch loading
        images = ImageIO.load_images_batch(image_paths)

        # Verify the number of loaded images
        assert len(images) == 3

        # Verify the shape of each image
        for img in images:
            assert img.shape == sample_image.shape

    def test_different_backends(self, temp_dir, sample_image):
        """Test different backends"""
        backends = ['opencv', 'pillow', 'skimage']

        for backend in backends:
            # Save the image
            image_path = temp_dir / f"test_{backend}.png"
            ImageIO.save_image(sample_image, image_path, backend=backend)

            # Load the image
            loaded = ImageIO.load_image(image_path, backend=backend)

            # Verify the shape
            assert loaded.shape == sample_image.shape
