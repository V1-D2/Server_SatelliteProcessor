"""
Image processing functions for satellite data
Based on user-provided polar image creation code
"""

import h5py
import numpy as np
import pyproj
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pathlib
from typing import List, Tuple, Optional
from PIL import Image


class ImageProcessor:
    """Processes satellite data into images"""

    def __init__(self):
        # EASE-Grid 2.0 parameters (same for North and South)
        self.PIXEL_SIZE_M = 10000.0  # 10 km pixels
        self.GRID_WIDTH = 1800  # Official grid width
        self.GRID_HEIGHT = 1800  # Official grid height

        # Map origin in projection coordinates
        self.MAP_ORIGIN_X = -9000000.0  # -9,000 km
        self.MAP_ORIGIN_Y = 9000000.0  # +9,000 km

        # Grid registration offset
        self.GRID_ORIGIN_COL = -0.5
        self.GRID_ORIGIN_ROW = -0.5

        # Set up projections for both poles
        self.ease2_north = pyproj.CRS.from_epsg(6931)  # EASE-Grid 2.0 North
        self.ease2_south = pyproj.CRS.from_epsg(6932)  # EASE-Grid 2.0 South
        self.wgs84 = pyproj.CRS.from_epsg(4326)  # WGS84

        # Initialize transformers (will be set based on pole)
        self.transformer = None

    def create_polar_image(self, h5_files: List[pathlib.Path],
                           orbit_type: str, pole: str = "N") -> Optional[np.ndarray]:
        """
        Create circular polar image from satellite data files

        Args:
            h5_files: List of HDF5 file paths
            orbit_type: 'A' for ascending, 'D' for descending
            pole: 'N' for north, 'S' for south

        Returns:
            Temperature array or None
        """
        # Set up transformer based on pole
        if pole == "N":
            self.transformer = pyproj.Transformer.from_crs(
                self.wgs84, self.ease2_north, always_xy=True
            )
        else:  # pole == "S"
            self.transformer = pyproj.Transformer.from_crs(
                self.wgs84, self.ease2_south, always_xy=True
            )

        # Create grids
        grid, weight, count, distance_from_pole = self._create_ease2_grid()

        # Process each file
        for idx, h5_path in enumerate(h5_files):
            try:
                self._add_swath_to_grid(
                    h5_path, grid, weight, count, idx, orbit_type, pole
                )
            except Exception as e:
                print(f"Error processing {h5_path.name}: {e}")
                continue

        # Finalize grid
        final_grid = self._finalize_grid(grid, weight, distance_from_pole)

        return final_grid

    def _create_ease2_grid(self):
        """Create empty grids for data accumulation"""
        grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.float64)
        weight = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.float64)
        count = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.int32)

        # Calculate distance from pole
        center_x = self.GRID_WIDTH // 2
        center_y = self.GRID_HEIGHT // 2
        y_indices, x_indices = np.meshgrid(
            range(self.GRID_HEIGHT), range(self.GRID_WIDTH), indexing='ij'
        )
        distance_from_pole = np.sqrt(
            (x_indices - center_x) ** 2 + (y_indices - center_y) ** 2
        )

        return grid, weight, count, distance_from_pole

    def _add_swath_to_grid(self, h5_path: pathlib.Path, grid: np.ndarray,
                           weight: np.ndarray, count: np.ndarray,
                           swath_idx: int, orbit_type: str, pole: str = "N"):
        """Add data from one swath file to the grid"""
        with h5py.File(h5_path, "r") as h5:
            # Extract temperature data
            var_name = "Brightness Temperature (36.5GHz,H)"
            if var_name not in h5:
                print(f"Variable {var_name} not found in {h5_path.name}")
                return

            raw = h5[var_name][:].astype(np.float64)

            # Get scale factor
            scale = 1.0
            if "SCALE FACTOR" in h5[var_name].attrs:
                scale = h5[var_name].attrs["SCALE FACTOR"]
                if isinstance(scale, np.ndarray):
                    scale = scale[0]

            # Apply scale factor and handle missing values
            tb = np.where(raw == 0, np.nan, raw * scale)

            # Get coordinates
            lat, lon = self._calculate_lat_lon_36ghz(h5)

            # Filter for correct hemisphere
            if pole == "N":
                hemisphere_mask = lat >= 0
            else:  # pole == "S"
                hemisphere_mask = lat <= 0

            if not np.any(hemisphere_mask):
                return

            # Transform to EASE-Grid 2.0
            x_ease2, y_ease2 = self._latlon_to_ease2(lat, lon)

            # Get grid bounds
            x_min, x_max, y_min, y_max = self._get_grid_bounds()

            # Valid data mask
            # Valid data mask
            valid_mask = (
                    hemisphere_mask &
                    ~np.isnan(tb) &
                    (x_ease2 >= x_min) & (x_ease2 <= x_max) &
                    (y_ease2 >= y_min) & (y_ease2 <= y_max) &
                    ~np.isnan(x_ease2) &
                    ~np.isnan(y_ease2)
            )

            if not np.any(valid_mask):
                return

            # Extract valid values
            x_vals = x_ease2[valid_mask]
            y_vals = y_ease2[valid_mask]
            tb_vals = tb[valid_mask]

            # Convert to pixel indices
            px_x, px_y = self._meters_to_pixels(x_vals, y_vals)

            # Check bounds
            valid_pixels = (
                    (px_x >= 0) & (px_x < self.GRID_WIDTH) &
                    (px_y >= 0) & (px_y < self.GRID_HEIGHT)
            )

            px_x = px_x[valid_pixels]
            px_y = px_y[valid_pixels]
            tb_vals = tb_vals[valid_pixels]

            # Accumulate data
            for i in range(len(tb_vals)):
                x_idx, y_idx = px_x[i], px_y[i]
                value = tb_vals[i]

                grid[y_idx, x_idx] += value
                weight[y_idx, x_idx] += 1.0
                count[y_idx, x_idx] += 1

    def _latlon_to_ease2(self, lat, lon):
        """Transform coordinates to EASE-Grid 2.0 (North or South based on current transformer)"""
        x, y = self.transformer.transform(lon, lat)
        x = np.where(np.isinf(x), np.nan, x)
        y = np.where(np.isinf(y), np.nan, y)
        return x, y

    def _meters_to_pixels(self, x_m, y_m):
        """Convert EASE-Grid 2.0 coordinates to pixel indices"""
        px_x = ((x_m - self.MAP_ORIGIN_X) / self.PIXEL_SIZE_M +
                self.GRID_ORIGIN_COL).astype(np.int32)
        px_y = ((self.MAP_ORIGIN_Y - y_m) / self.PIXEL_SIZE_M +
                self.GRID_ORIGIN_ROW).astype(np.int32)
        return px_x, px_y

    def _get_grid_bounds(self):
        """Get grid bounds in meters"""
        x_min = self.MAP_ORIGIN_X
        x_max = self.MAP_ORIGIN_X + self.GRID_WIDTH * self.PIXEL_SIZE_M
        y_min = self.MAP_ORIGIN_Y - self.GRID_HEIGHT * self.PIXEL_SIZE_M
        y_max = self.MAP_ORIGIN_Y
        return x_min, x_max, y_min, y_max

    def _calculate_lat_lon_36ghz(self, h5):
        """Calculate lat/lon for 36.5 GHz channel"""
        lat_89 = None
        lon_89 = None

        # Find 89 GHz coordinates
        for suffix in ["89A", "89B"]:
            lat_key = f"Latitude of Observation Point for {suffix}"
            lon_key = f"Longitude of Observation Point for {suffix}"

            if lat_key in h5 and lon_key in h5:
                lat_89 = h5[lat_key][:]
                lon_89 = h5[lon_key][:]
                break

        if lat_89 is None:
            raise ValueError("Coordinates not found in file!")

        # Downsample if needed
        if lat_89.shape[1] == 486:  # High resolution
            lat_36 = lat_89[:, ::2]
            lon_36 = lon_89[:, ::2]
        else:
            lat_36 = lat_89
            lon_36 = lon_89

        return lat_36, lon_36

    def _finalize_grid(self, grid, weight, distance_from_pole, apply_filling=True):
        """Finalize grid and optionally fill holes"""
        final_grid = np.full(grid.shape, np.nan, dtype=np.float32)
        valid_mask = weight > 0

        if not np.any(valid_mask):
            print("ERROR: No valid data for finalization!")
            return final_grid

        final_grid[valid_mask] = (grid[valid_mask] / weight[valid_mask]).astype(np.float32)

        if apply_filling:
            final_grid = self._smart_fill_holes(final_grid, distance_from_pole)

        return final_grid

    def _smart_fill_holes(self, data, distance_from_pole):
        """Fill holes in data using weighted interpolation"""
        filled_data = data.copy()
        rows, cols = data.shape

        empty_mask = np.isnan(data)
        initial_holes = np.sum(empty_mask)

        if initial_holes == 0:
            return filled_data

        # Parameters for adaptive radius
        MIN_RADIUS = 2
        MAX_RADIUS = 6
        DISTANCE_SCALE = 400
        COVERAGE_THRESHOLD = 0.3

        filled_count = 0

        for y in range(rows):
            for x in range(cols):
                if not empty_mask[y, x]:
                    continue

                dist_from_pole = distance_from_pole[y, x]

                # Adaptive radius
                radius_factor = min(dist_from_pole / DISTANCE_SCALE, 1.0)
                search_radius = int(MIN_RADIUS + radius_factor * (MAX_RADIUS - MIN_RADIUS))

                # Define neighborhood
                y_min = max(0, y - search_radius)
                y_max = min(rows, y + search_radius + 1)
                x_min = max(0, x - search_radius)
                x_max = min(cols, x + search_radius + 1)

                neighborhood = data[y_min:y_max, x_min:x_max]
                valid_mask = ~np.isnan(neighborhood)
                valid_count = np.sum(valid_mask)
                total_count = neighborhood.size

                coverage = valid_count / total_count if total_count > 0 else 0

                if coverage >= COVERAGE_THRESHOLD and valid_count >= 3:
                    # Get valid coordinates
                    valid_y, valid_x = np.where(valid_mask)
                    valid_y += y_min
                    valid_x += x_min

                    # Calculate weights based on distance
                    distances = np.sqrt((valid_y - y) ** 2 + (valid_x - x) ** 2)
                    weights = 1.0 / ((distances + 0.1) ** 4)
                    weights /= weights.sum()

                    # Weighted average
                    valid_values = data[valid_y, valid_x]
                    filled_value = np.sum(valid_values * weights)
                    filled_data[y, x] = filled_value
                    filled_count += 1

        print(f"Filled {filled_count} holes out of {initial_holes}")
        return filled_data

    def save_color_image(self, data: np.ndarray, output_path: pathlib.Path):
        """Save data as color image using turbo colormap"""
        # Handle NaN values
        valid_mask = ~np.isnan(data)
        if not np.any(valid_mask):
            print("No valid data to save")
            return

        # Get data range
        vmin = np.nanmin(data)
        vmax = np.nanmax(data)

        # Create figure without margins
        h, w = data.shape
        dpi = 100
        fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        ax = plt.axes([0, 0, 1, 1])
        ax.axis('off')

        # Set black background for NaN values
        import matplotlib.cm as cm
        from matplotlib.colors import ListedColormap

        # Get turbo colormap
        turbo = cm.get_cmap('turbo')
        # Set bad values (NaN) to black
        turbo.set_bad(color='black')

        # Plot with turbo colormap
        im = ax.imshow(
            data,
            cmap=turbo,
            vmin=vmin,
            vmax=vmax,
            interpolation='nearest',
            aspect='auto'
        )

        # Save
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close()

    def save_grayscale_image(self, data: np.ndarray, output_path: pathlib.Path):
        """Save data as grayscale image"""
        # Handle NaN values
        valid_mask = ~np.isnan(data)
        if not np.any(valid_mask):
            print("No valid data to save")
            return

        # Normalize data to 0-255
        data_normalized = data.copy()
        data_normalized[~valid_mask] = np.nanmin(data)  # Fill NaN with min

        # Normalize
        vmin = np.nanmin(data_normalized)
        vmax = np.nanmax(data_normalized)

        if vmax > vmin:
            data_normalized = (data_normalized - vmin) / (vmax - vmin) * 255
        else:
            data_normalized = np.zeros_like(data_normalized)

        # Convert to uint8
        data_uint8 = data_normalized.astype(np.uint8)

        # Save using PIL
        img = Image.fromarray(data_uint8, mode='L')
        img.save(output_path)

    def save_viridis_image(self, data: np.ndarray, output_path: pathlib.Path):
        """Save data as viridis colormap image"""
        # Handle NaN values
        valid_mask = ~np.isnan(data)
        if not np.any(valid_mask):
            print("No valid data to save")
            return

        # Get data range
        vmin = np.nanmin(data)
        vmax = np.nanmax(data)

        # Create figure without margins
        h, w = data.shape
        dpi = 100
        fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        ax = plt.axes([0, 0, 1, 1])
        ax.axis('off')

        # Plot with viridis colormap
        im = ax.imshow(
            data,
            cmap='viridis',
            vmin=vmin,
            vmax=vmax,
            interpolation='nearest',
            aspect='auto'
        )

        # Save
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close()

    def save_color_image_percentile(self, data: np.ndarray, output_path: pathlib.Path):
        """Save data as color image using turbo colormap with percentile filtering"""
        # Handle NaN values
        valid_mask = ~np.isnan(data)
        if not np.any(valid_mask):
            print("No valid data to save")
            return

        # Use percentile for better contrast
        p_low, p_high = 1, 99
        vmin = np.nanpercentile(data, p_low)
        vmax = np.nanpercentile(data, p_high)

        # Create figure without margins
        h, w = data.shape
        dpi = 100
        fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        ax = plt.axes([0, 0, 1, 1])
        ax.axis('off')

        # Set black background for NaN values
        import matplotlib.cm as cm
        from matplotlib.colors import ListedColormap

        # Get turbo colormap
        turbo = cm.get_cmap('turbo')
        # Set bad values (NaN) to black
        turbo.set_bad(color='black')

        # Plot with turbo colormap
        im = ax.imshow(
            data,
            cmap=turbo,
            vmin=vmin,
            vmax=vmax,
            interpolation='nearest',
            aspect='auto'
        )
        # Save
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close()

    def save_grayscale_image_percentile(self, data: np.ndarray, output_path: pathlib.Path):
        """Save data as grayscale image with percentile filtering"""
        # Handle NaN values
        valid_mask = ~np.isnan(data)
        if not np.any(valid_mask):
            print("No valid data to save")
            return

        # Normalize data to 0-255
        data_normalized = data.copy()
        data_normalized[~valid_mask] = np.nanmin(data)  # Fill NaN with min

        # Use percentile
        p_low, p_high = 1, 99
        vmin = np.nanpercentile(data_normalized, p_low)
        vmax = np.nanpercentile(data_normalized, p_high)

        if vmax > vmin:
            # Clip to percentile range first
            data_normalized = np.clip(data_normalized, vmin, vmax)
            # Then normalize
            data_normalized = (data_normalized - vmin) / (vmax - vmin) * 255
        else:
            data_normalized = np.zeros_like(data_normalized)

        # Convert to uint8
        data_uint8 = data_normalized.astype(np.uint8)

        # Save using PIL
        img = Image.fromarray(data_uint8, mode='L')
        img.save(output_path)

    def tensor2img(self, tensor_list):
        """
        Convert tensor to grayscale image (placeholder for user's method)
        This mimics the user's tensor2img function
        """
        # For now, just normalize to grayscale
        # User can replace this with their specific implementation
        if len(tensor_list) > 0:
            data = tensor_list[0]
            return self._normalize_to_uint8(data)
        return None

    def _normalize_to_uint8(self, data):
        """Normalize data to uint8 range"""
        valid_mask = ~np.isnan(data)
        if not np.any(valid_mask):
            return np.zeros_like(data, dtype=np.uint8)

        vmin = np.nanmin(data)
        vmax = np.nanmax(data)

        if vmax > vmin:
            normalized = (data - vmin) / (vmax - vmin) * 255
            normalized[~valid_mask] = 0
            return normalized.astype(np.uint8)
        else:
            return np.zeros_like(data, dtype=np.uint8)