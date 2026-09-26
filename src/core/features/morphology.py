"""
Cell morphology feature extraction.

Extract geometric and morphological features for individual cells.
"""
import numpy as np
import pandas as pd
from skimage import measure
from typing import Dict, List
from loguru import logger


def extract_cell_features(mask: np.ndarray, pixel_size: float = 1.0, min_area: int = 100) -> pd.DataFrame:
    """
    Extract cell morphology features.

    Args:
        mask: Segmentation mask with a unique label for each cell.
        pixel_size: Pixel size in micrometers per pixel for conversion to physical units.
        min_area: Minimum cell area in pixels; smaller cells are excluded.

    Returns:
        DataFrame containing features for all retained cells.
    """
    # Extract features using regionprops.
    regions = measure.regionprops(mask)

    if len(regions) == 0:
        logger.warning("No cells detected in mask")
        return pd.DataFrame()

    features_list = []
    sequential_id = 0  # Sequential ID counter.

    for region in regions:
        # Exclude cells below the minimum area.
        if region.area < min_area:
            continue

        sequential_id += 1  # Increment the sequential ID.

        features = {
            # Cell identifiers: sequential ID and original mask label.
            'sequential_id': sequential_id,  # Sequential ID (1, 2, 3, ...).
            'cell_id': region.label,  # Original mask label, which may be nonconsecutive.

            # Position features.
            'centroid_y': region.centroid[0],
            'centroid_x': region.centroid[1],

            # Area and perimeter features.
            'area_pixels': region.area,
            'area_um2': region.area * (pixel_size ** 2),
            'perimeter_pixels': region.perimeter,
            'perimeter_um': region.perimeter * pixel_size,

            # Shape features.
            'major_axis_length': region.major_axis_length * pixel_size,
            'minor_axis_length': region.minor_axis_length * pixel_size,
            'eccentricity': region.eccentricity,
            'solidity': region.solidity,
            'extent': region.extent,

            # Compute circularity (4*pi*area/perimeter^2).
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

        features_list.append(features)

    df = pd.DataFrame(features_list)

    logger.info(f"Extracted features for {len(df)} cells")

    return df


def get_feature_statistics(df: pd.DataFrame) -> Dict:
    """
    Compute feature summary statistics.

    Args:
        df: Feature DataFrame.

    Returns:
        Dictionary of summary statistics.
    """
    if df.empty:
        return {}

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['cell_id', 'centroid_x', 'centroid_y',
                                                                 'bbox_min_row', 'bbox_min_col',
                                                                 'bbox_max_row', 'bbox_max_col']]

    stats = {}
    for col in numeric_cols:
        stats[col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'median': df[col].median()
        }

    return stats


def filter_cells_by_features(df: pd.DataFrame,
                             min_area: float = None,
                             max_area: float = None,
                             min_circularity: float = None,
                             max_circularity: float = None) -> pd.DataFrame:
    """
    Filter cells by feature values.

    Args:
        df: Feature DataFrame.
        min_area: Minimum area in square micrometers.
        max_area: Maximum area in square micrometers.
        min_circularity: Minimum circularity.
        max_circularity: Maximum circularity.

    Returns:
        Filtered DataFrame.
    """
    filtered_df = df.copy()

    if min_area is not None:
        filtered_df = filtered_df[filtered_df['area_um2'] >= min_area]

    if max_area is not None:
        filtered_df = filtered_df[filtered_df['area_um2'] <= max_area]

    if min_circularity is not None:
        filtered_df = filtered_df[filtered_df['circularity'] >= min_circularity]

    if max_circularity is not None:
        filtered_df = filtered_df[filtered_df['circularity'] <= max_circularity]

    logger.info(f"Filtered from {len(df)} to {len(filtered_df)} cells")

    return filtered_df
