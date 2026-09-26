"""
Advanced cell morphology feature extraction.

Provides additional morphological, texture, and intensity features.
"""
import numpy as np
import pandas as pd
from skimage import measure, feature
from skimage.measure import moments_hu
from scipy import ndimage
from scipy.stats import skew, kurtosis, entropy
from typing import Dict, List, Optional
from loguru import logger


def extract_hu_moments(region) -> Dict[str, float]:
    """
    Extract seven Hu moment shape descriptors.

    Args:
        region: A skimage regionprops object.

    Returns:
        Dictionary containing seven Hu moments.
    """
    # Get central moments for Hu moment computation.
    hu = moments_hu(region.moments_central)

    # Log-transform Hu moments for analysis.
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    return {
        'hu_moment_1': hu_log[0],
        'hu_moment_2': hu_log[1],
        'hu_moment_3': hu_log[2],
        'hu_moment_4': hu_log[3],
        'hu_moment_5': hu_log[4],
        'hu_moment_6': hu_log[5],
        'hu_moment_7': hu_log[6],
    }


def extract_intensity_features(region, image: np.ndarray) -> Dict[str, float]:
    """
    Extract intensity statistics.

    Args:
        region: A skimage regionprops object.
        image: Original grayscale image.

    Returns:
        Dictionary of intensity features.
    """
    # Extract pixel intensities within the cell region.
    cell_pixels = image[region.coords[:, 0], region.coords[:, 1]]

    # Basic statistics.
    mean_intensity = np.mean(cell_pixels)
    std_intensity = np.std(cell_pixels)
    min_intensity = np.min(cell_pixels)
    max_intensity = np.max(cell_pixels)
    median_intensity = np.median(cell_pixels)

    # Higher-order statistics.
    skewness = skew(cell_pixels)
    kurt = kurtosis(cell_pixels)

    # Compute information entropy.
    hist, _ = np.histogram(cell_pixels, bins=256, range=(0, 256))
    hist = hist / hist.sum()  # Normalize.
    ent = entropy(hist + 1e-10)  # Avoid log(0).

    # Intensity range and contrast.
    intensity_range = max_intensity - min_intensity

    return {
        'intensity_mean': mean_intensity,
        'intensity_std': std_intensity,
        'intensity_min': min_intensity,
        'intensity_max': max_intensity,
        'intensity_median': median_intensity,
        'intensity_range': intensity_range,
        'intensity_skewness': skewness,
        'intensity_kurtosis': kurt,
        'intensity_entropy': ent,
    }


def extract_boundary_features(region, pixel_size: float = 1.0) -> Dict[str, float]:
    """
    Extract boundary complexity features.

    Args:
        region: A skimage regionprops object.
        pixel_size: Pixel size in micrometers per pixel.

    Returns:
        Dictionary of boundary features.
    """
    # Boundary roughness = perimeter^2 / (4*pi*area).
    # A perfect circle has roughness 1; rougher boundaries yield larger values.
    roughness = (region.perimeter ** 2) / (4 * np.pi * region.area) if region.area > 0 else 0

    # Compactness = area / convex-hull area.
    compactness = region.solidity

    # Concavity estimate = estimated convex-hull perimeter - perimeter.
    convex_perimeter = region.perimeter / region.solidity if region.solidity > 0 else region.perimeter
    concavity = (convex_perimeter - region.perimeter) * pixel_size

    # Shape factor = 4*pi*area/perimeter^2, equivalent to circularity.
    shape_factor = (4 * np.pi * region.area) / (region.perimeter ** 2) if region.perimeter > 0 else 0

    return {
        'boundary_roughness': roughness,
        'boundary_compactness': compactness,
        'boundary_concavity': concavity,
        'shape_factor': shape_factor,
    }


def extract_texture_features_glcm(region, image: np.ndarray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]) -> Dict[str, float]:
    """
    Extract Haralick texture features from gray-level co-occurrence matrices (GLCMs).

    Args:
        region: A skimage regionprops object.
        image: Original grayscale image.
        distances: List of pixel distances used to compute GLCMs.
        angles: List of angles used to compute GLCMs.

    Returns:
        Dictionary of texture features.
    """
    # Extract the image bounding box of the cell.
    minr, minc, maxr, maxc = region.bbox
    cell_image = image[minr:maxr, minc:maxc]
    cell_mask = region.image

    # Retain only pixels within the cell region.
    cell_image_masked = cell_image * cell_mask

    # Normalize to [0, 255].
    if cell_image_masked.max() > 0:
        cell_image_normalized = ((cell_image_masked - cell_image_masked.min()) /
                                 (cell_image_masked.max() - cell_image_masked.min()) * 255).astype(np.uint8)
    else:
        cell_image_normalized = cell_image_masked.astype(np.uint8)

    try:
        # Compute the GLCMs.
        glcm = feature.graycomatrix(cell_image_normalized, distances=distances, angles=angles,
                                    levels=256, symmetric=True, normed=True)

        # Extract Haralick features.
        contrast = feature.graycoprops(glcm, 'contrast').mean()
        dissimilarity = feature.graycoprops(glcm, 'dissimilarity').mean()
        homogeneity = feature.graycoprops(glcm, 'homogeneity').mean()
        energy = feature.graycoprops(glcm, 'energy').mean()
        correlation = feature.graycoprops(glcm, 'correlation').mean()
        asm = feature.graycoprops(glcm, 'ASM').mean()

        return {
            'texture_contrast': contrast,
            'texture_dissimilarity': dissimilarity,
            'texture_homogeneity': homogeneity,
            'texture_energy': energy,
            'texture_correlation': correlation,
            'texture_asm': asm,
        }
    except Exception as e:
        logger.warning(f"Failed to extract GLCM features: {e}")
        return {
            'texture_contrast': 0,
            'texture_dissimilarity': 0,
            'texture_homogeneity': 0,
            'texture_energy': 0,
            'texture_correlation': 0,
            'texture_asm': 0,
        }


def calculate_fractal_dimension(region) -> float:
    """
    Estimate the fractal dimension of the boundary using box counting.

    Args:
        region: A skimage regionprops object.

    Returns:
        Fractal dimension.
    """
    try:
        # Get the binary cell image.
        binary_image = region.image.astype(bool)

        # Extract the boundary.
        from scipy import ndimage
        boundary = binary_image ^ ndimage.binary_erosion(binary_image)

        # Box-counting algorithm.
        def boxcount(image, k):
            S = np.add.reduceat(
                np.add.reduceat(image, np.arange(0, image.shape[0], k), axis=0),
                np.arange(0, image.shape[1], k), axis=1)
            return len(np.where(S > 0)[0])

        # Count occupied boxes at different scales.
        scales = np.array([2, 4, 8, 16])
        scales = scales[scales < min(boundary.shape)]

        if len(scales) < 2:
            return 0

        counts = []
        for scale in scales:
            counts.append(boxcount(boundary, scale))

        # Fit log(N) against log(scale), then negate the slope.
        coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
        fractal_dim = -coeffs[0]

        return fractal_dim
    except Exception as e:
        logger.warning(f"Failed to calculate fractal dimension: {e}")
        return 0


def extract_advanced_shape_features(region, pixel_size: float = 1.0) -> Dict[str, float]:
    """
    Extract advanced shape features.

    Args:
        region: A skimage regionprops object.
        pixel_size: Pixel size in micrometers per pixel.

    Returns:
        Dictionary of advanced shape features.
    """
    # Ellipticity = minor axis / major axis.
    ellipticity = region.minor_axis_length / region.major_axis_length if region.major_axis_length > 0 else 0

    # Elongation = 1 - ellipticity.
    elongation = 1 - ellipticity

    # Rectangularity = area / bounding-box area.
    bbox_area = (region.bbox[2] - region.bbox[0]) * (region.bbox[3] - region.bbox[1])
    rectangularity = region.area / bbox_area if bbox_area > 0 else 0

    # Equivalent ellipse area.
    equivalent_ellipse_area = np.pi * region.major_axis_length * region.minor_axis_length / 4

    # Fractal dimension.
    fractal_dim = calculate_fractal_dimension(region)

    return {
        'ellipticity': ellipticity,
        'elongation': elongation,
        'rectangularity': rectangularity,
        'equivalent_ellipse_area': equivalent_ellipse_area * (pixel_size ** 2),
        'fractal_dimension': fractal_dim,
    }


def extract_advanced_cell_features(
    mask: np.ndarray,
    image: Optional[np.ndarray] = None,
    pixel_size: float = 1.0,
    min_area: int = 100,
    include_hu_moments: bool = True,
    include_intensity: bool = True,
    include_texture: bool = True,
    include_boundary: bool = True,
    include_advanced_shape: bool = True
) -> pd.DataFrame:
    """
    Extract the complete set of advanced cell morphology features.

    Args:
        mask: Segmentation mask with a unique label for each cell.
        image: Original grayscale image for intensity and texture features.
        pixel_size: Pixel size in micrometers per pixel.
        min_area: Minimum cell area in pixels.
        include_hu_moments: Include Hu moment features.
        include_intensity: Include intensity features; requires image.
        include_texture: Include texture features; requires image.
        include_boundary: Include boundary features.
        include_advanced_shape: Include advanced shape features.

    Returns:
        DataFrame containing all requested advanced features.
    """
    # Extract basic region information using regionprops.
    if image is not None:
        regions = measure.regionprops(mask, intensity_image=image)
    else:
        regions = measure.regionprops(mask)

    if len(regions) == 0:
        logger.warning("No cells detected in mask")
        return pd.DataFrame()

    features_list = []
    sequential_id = 0

    for region in regions:
        # Exclude cells below the minimum area.
        if region.area < min_area:
            continue

        sequential_id += 1

        # Basic features, including standard morphological descriptors.
        features = {
            'sequential_id': sequential_id,
            'cell_id': region.label,
            'centroid_y': region.centroid[0],
            'centroid_x': region.centroid[1],
            'area_pixels': region.area,
            'area_um2': region.area * (pixel_size ** 2),
            'perimeter_pixels': region.perimeter,
            'perimeter_um': region.perimeter * pixel_size,

            # Basic shape features.
            'major_axis_length': region.major_axis_length * pixel_size,
            'minor_axis_length': region.minor_axis_length * pixel_size,
            'eccentricity': region.eccentricity,
            'solidity': region.solidity,
            'extent': region.extent,

            # Compute circularity.
            'circularity': (4 * np.pi * region.area) / (region.perimeter ** 2) if region.perimeter > 0 else 0,

            # Aspect ratio.
            'aspect_ratio': region.major_axis_length / region.minor_axis_length if region.minor_axis_length > 0 else 0,

            # Equivalent diameter.
            'equivalent_diameter_pixels': region.equivalent_diameter,
            'equivalent_diameter_um': region.equivalent_diameter * pixel_size,

            # Bounding box.
            'bbox_min_row': region.bbox[0],
            'bbox_min_col': region.bbox[1],
            'bbox_max_row': region.bbox[2],
            'bbox_max_col': region.bbox[3],
        }

        # Hu moment features.
        if include_hu_moments:
            try:
                hu_features = extract_hu_moments(region)
                features.update(hu_features)
            except Exception as e:
                logger.warning(f"Failed to extract Hu moments for cell {region.label}: {e}")

        # Intensity features.
        if include_intensity and image is not None:
            try:
                intensity_features = extract_intensity_features(region, image)
                features.update(intensity_features)
            except Exception as e:
                logger.warning(f"Failed to extract intensity features for cell {region.label}: {e}")

        # Texture features.
        if include_texture and image is not None:
            try:
                texture_features = extract_texture_features_glcm(region, image)
                features.update(texture_features)
            except Exception as e:
                logger.warning(f"Failed to extract texture features for cell {region.label}: {e}")

        # Boundary features.
        if include_boundary:
            try:
                boundary_features = extract_boundary_features(region, pixel_size)
                features.update(boundary_features)
            except Exception as e:
                logger.warning(f"Failed to extract boundary features for cell {region.label}: {e}")

        # Advanced shape features.
        if include_advanced_shape:
            try:
                shape_features = extract_advanced_shape_features(region, pixel_size)
                features.update(shape_features)
            except Exception as e:
                logger.warning(f"Failed to extract advanced shape features for cell {region.label}: {e}")

        features_list.append(features)

    df = pd.DataFrame(features_list)
    logger.info(f"Extracted advanced features for {len(df)} cells")

    return df
