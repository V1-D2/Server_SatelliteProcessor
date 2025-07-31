"""
Temperature data preprocessing adapted for SatelliteProcessor
"""

import numpy as np
import torch
from typing import Tuple


class TemperatureDataPreprocessor:
    """Preprocessor for temperature data"""

    def __init__(self, target_height: int = 2000, target_width: int = 220):
        self.target_height = target_height
        self.target_width = target_width

    def normalize_temperature(self, temp_array: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Normalize temperature to [0, 1] range

        Returns:
            Normalized array, min value, max value
        """
        # Handle NaN values
        mask = np.isnan(temp_array)
        if mask.any():
            mean_val = np.nanmean(temp_array)
            temp_array = np.where(mask, mean_val, temp_array)

        min_temp = np.min(temp_array)
        max_temp = np.max(temp_array)

        if max_temp > min_temp:
            normalized = (temp_array - min_temp) / (max_temp - min_temp)
        else:
            normalized = np.zeros_like(temp_array)

        return normalized.astype(np.float32), min_temp, max_temp

    def denormalize_temperature(self, normalized: np.ndarray,
                                original_min: float, original_max: float) -> np.ndarray:
        """Denormalize from [0, 1] back to temperature values"""
        return normalized * (original_max - original_min) + original_min

    def crop_or_pad(self, temp_array: np.ndarray) -> np.ndarray:
        """Crop or pad to target size if needed"""
        h, w = temp_array.shape

        # For SatelliteProcessor, we typically don't need to resize
        # but keep this method for compatibility
        return temp_array