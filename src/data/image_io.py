"""
Image input/output utilities.

Provides image loading, saving, and basic image operations.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple, Optional
from PIL import Image
import skimage.io as skio

from ..core.utils.logger import get_logger

logger = get_logger(__name__)


class ImageIO:
    """Image input/output methods."""

    # Supported image formats.
    SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']

    @staticmethod
    def load_image(
        image_path: Union[str, Path],
        grayscale: bool = False,
        backend: str = 'opencv'
    ) -> np.ndarray:
        """
        Load an image.

        Args:
            image_path: Path to the image file.
            grayscale: Convert the image to grayscale.
            backend: Image backend ('opencv', 'pillow', 'skimage').

        Returns:
            Image array (H, W, C) or (H, W).

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the image format is unsupported.
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if image_path.suffix.lower() not in ImageIO.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported image format: {image_path.suffix}")

        try:
            if backend == 'opencv':
                # OpenCV reads color images in BGR order by default.
                if grayscale:
                    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                else:
                    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    # Convert to RGB.
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            elif backend == 'pillow':
                image = Image.open(image_path)
                if grayscale:
                    image = image.convert('L')
                else:
                    image = image.convert('RGB')
                image = np.array(image)

            elif backend == 'skimage':
                image = skio.imread(str(image_path), as_gray=grayscale)
                if not grayscale and image.ndim == 2:
                    # Expand grayscale to three channels when color output is requested.
                    image = np.stack([image] * 3, axis=-1)

            else:
                raise ValueError(f"Unknown backend: {backend}")

            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            logger.debug(f"Loaded image: {image_path}, shape: {image.shape}")
            return image

        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise

    @staticmethod
    def save_image(
        image: np.ndarray,
        output_path: Union[str, Path],
        backend: str = 'opencv'
    ) -> None:
        """
        Save an image.

        Args:
            image: Image array.
            output_path: Output file path.
            backend: Image backend ('opencv', 'pillow', 'skimage').
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if backend == 'opencv':
                # Convert RGB images to BGR.
                if image.ndim == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), image)

            elif backend == 'pillow':
                if image.ndim == 2:
                    mode = 'L'
                elif image.shape[2] == 3:
                    mode = 'RGB'
                elif image.shape[2] == 4:
                    mode = 'RGBA'
                else:
                    raise ValueError(f"Unsupported image shape: {image.shape}")

                pil_image = Image.fromarray(image, mode=mode)
                pil_image.save(output_path)

            elif backend == 'skimage':
                skio.imsave(str(output_path), image)

            else:
                raise ValueError(f"Unknown backend: {backend}")

            logger.debug(f"Saved image to: {output_path}")

        except Exception as e:
            logger.error(f"Error saving image to {output_path}: {e}")
            raise

    @staticmethod
    def get_image_info(image_path: Union[str, Path]) -> dict:
        """
        Get image metadata.

        Args:
            image_path: Path to the image file.

        Returns:
            Dictionary of image metadata.
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Use PIL to read basic metadata without loading the full image.
        with Image.open(image_path) as img:
            info = {
                'path': str(image_path),
                'filename': image_path.name,
                'format': img.format,
                'mode': img.mode,
                'size': img.size,  # (width, height)
                'width': img.width,
                'height': img.height,
            }

        return info

    @staticmethod
    def load_images_batch(
        image_paths: List[Union[str, Path]],
        grayscale: bool = False,
        backend: str = 'opencv'
    ) -> List[np.ndarray]:
        """
        Load multiple images.

        Args:
            image_paths: List of image file paths.
            grayscale: Convert the image to grayscale.
            backend: Image backend.

        Returns:
            List of image arrays.
        """
        images = []
        for path in image_paths:
            try:
                image = ImageIO.load_image(path, grayscale=grayscale, backend=backend)
                images.append(image)
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
                continue

        logger.info(f"Loaded {len(images)}/{len(image_paths)} images")
        return images


def load_image(image_path: Union[str, Path], grayscale: bool = False) -> np.ndarray:
    """
    Convenience wrapper for loading an image.

    Args:
        image_path: Path to the image file.
        grayscale: Convert the image to grayscale.

    Returns:
        Image array.
    """
    return ImageIO.load_image(image_path, grayscale=grayscale)


def save_image(image: np.ndarray, output_path: Union[str, Path]) -> None:
    """
    Convenience wrapper for saving an image.

    Args:
        image: Image array.
        output_path: Output file path.
    """
    ImageIO.save_image(image, output_path)
