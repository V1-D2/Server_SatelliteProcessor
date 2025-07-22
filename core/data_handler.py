"""
Data handler for temperature extraction and array operations
"""

import h5py
import numpy as np
import pathlib
from typing import Tuple, Optional


class DataHandler:
    """Handles data extraction and array operations"""

    def __init__(self):
        self.var_name = "Brightness Temperature (36.5GHz,H)"

    def extract_temperature_data(self, h5_path: pathlib.Path) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Extract temperature data and scale factor from HDF5 file

        Args:
            h5_path: Path to HDF5 file

        Returns:
            Tuple of (temperature_array, scale_factor) or (None, None)
        """
        try:
            with h5py.File(h5_path, "r") as h5:
                # Check if variable exists
                if self.var_name not in h5:
                    print(f"Variable '{self.var_name}' not found in {h5_path.name}")
                    available_vars = list(h5.keys())
                    print(f"Available variables: {available_vars}")
                    return None, None

                # Get raw data
                raw_data = h5[self.var_name][:]

                # Get scale factor
                scale_factor = 1.0
                if "SCALE FACTOR" in h5[self.var_name].attrs:
                    scale_factor = h5[self.var_name].attrs["SCALE FACTOR"]
                    if isinstance(scale_factor, np.ndarray):
                        scale_factor = scale_factor[0]

                # Apply scale factor and handle missing values
                # Convert 0 values to NaN
                temp_data = np.where(raw_data == 0, np.nan, raw_data * scale_factor)

                # Verify we have valid data
                valid_count = np.sum(~np.isnan(temp_data))
                if valid_count == 0:
                    print(f"No valid temperature data in {h5_path.name}")
                    return None, None

                print(f"Extracted temperature data: shape={temp_data.shape}, "
                      f"valid_pixels={valid_count}, "
                      f"scale_factor={scale_factor}")

                return temp_data, scale_factor

        except Exception as e:
            print(f"Error extracting temperature data from {h5_path.name}: {e}")
            return None, None

    def extract_metadata(self, h5_path: pathlib.Path) -> dict:
        """
        Extract metadata from HDF5 file

        Args:
            h5_path: Path to HDF5 file

        Returns:
            Dictionary with metadata
        """
        metadata = {}

        try:
            with h5py.File(h5_path, "r") as h5:
                # Get basic info
                metadata['filename'] = h5_path.name
                metadata['variables'] = list(h5.keys())

                # Get attributes
                if hasattr(h5, 'attrs'):
                    metadata['global_attrs'] = dict(h5.attrs)

                # Get specific variable info
                if self.var_name in h5:
                    var = h5[self.var_name]
                    metadata['shape'] = var.shape
                    metadata['dtype'] = str(var.dtype)

                    # Variable attributes
                    if hasattr(var, 'attrs'):
                        metadata['var_attrs'] = dict(var.attrs)

                # Try to determine orbit type from filename
                orbit_type = "Unknown"
                try:
                    parts = h5_path.stem.split("_")
                    if len(parts) >= 3:
                        ad_flag = parts[2][-1]
                        if ad_flag in ["A", "D"]:
                            orbit_type = ad_flag
                except:
                    pass

                metadata['orbit_type'] = orbit_type

        except Exception as e:
            print(f"Error extracting metadata: {e}")

        return metadata

    def save_temperature_array(self, data: np.ndarray, output_path: pathlib.Path):
        """
        Save temperature array as NPZ file

        Args:
            data: Temperature array with NaN for missing values
            output_path: Path to save NPZ file
        """
        try:
            # Calculate statistics
            valid_mask = ~np.isnan(data)
            stats = {
                'shape': data.shape,
                'valid_pixels': np.sum(valid_mask),
                'min_temp': np.nanmin(data) if np.any(valid_mask) else None,
                'max_temp': np.nanmax(data) if np.any(valid_mask) else None,
                'mean_temp': np.nanmean(data) if np.any(valid_mask) else None,
                'coverage_percent': 100 * np.sum(valid_mask) / data.size
            }

            # Save with compression
            np.savez_compressed(
                output_path,
                temperature=data,
                stats=stats
            )

            # Report file size
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"Saved temperature array: {output_path.name} ({file_size_mb:.2f} MB)")
            print(f"  Shape: {stats['shape']}")
            print(f"  Valid pixels: {stats['valid_pixels']} ({stats['coverage_percent']:.1f}%)")
            if stats['min_temp'] is not None:
                print(f"  Temperature range: {stats['min_temp']:.1f} - {stats['max_temp']:.1f} K")

        except Exception as e:
            print(f"Error saving temperature array: {e}")
            raise

    def load_temperature_array(self, npz_path: pathlib.Path) -> Optional[np.ndarray]:
        """
        Load temperature array from NPZ file

        Args:
            npz_path: Path to NPZ file

        Returns:
            Temperature array or None
        """
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                if 'temperature' in data:
                    temp_array = data['temperature']

                    # Print stats if available
                    if 'stats' in data:
                        stats = data['stats'].item()
                        print(f"Loaded temperature array from {npz_path.name}")
                        print(f"  Shape: {stats.get('shape', temp_array.shape)}")
                        print(f"  Coverage: {stats.get('coverage_percent', 'Unknown')}%")

                    return temp_array
                else:
                    print(f"No temperature data found in {npz_path.name}")
                    return None

        except Exception as e:
            print(f"Error loading temperature array: {e}")
            return None

    def combine_temperature_arrays(self, arrays: list) -> np.ndarray:
        """
        Combine multiple temperature arrays

        Args:
            arrays: List of temperature arrays

        Returns:
            Combined array
        """
        if not arrays:
            return None

        if len(arrays) == 1:
            return arrays[0]

        # Ensure all arrays have the same shape
        shape = arrays[0].shape
        for arr in arrays[1:]:
            if arr.shape != shape:
                raise ValueError(f"Array shape mismatch: {arr.shape} != {shape}")

        # Create combined array
        combined = np.full(shape, np.nan)
        count = np.zeros(shape)

        # Average overlapping values
        for arr in arrays:
            valid_mask = ~np.isnan(arr)
            combined[valid_mask] = np.nansum([combined[valid_mask], arr[valid_mask]], axis=0)
            count[valid_mask] += 1

        # Compute average
        mask = count > 0
        combined[mask] /= count[mask]

        return combined