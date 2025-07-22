"""
GPORTAL API client for downloading AMSR-2 data
Based on user-provided code
"""

import pathlib
import datetime as dt
import tqdm
import gportal
import os
from typing import List, Dict, Optional


class GPortalClient:
    """Client for interacting with GPORTAL API"""

    def __init__(self, auth_manager):
        self.auth_manager = auth_manager
        self._setup_credentials()
        self._setup_dataset()

    def _setup_credentials(self):
        """Set up gportal credentials"""
        username, password = self.auth_manager.get_credentials()
        if username and password:
            gportal.username = username
            gportal.password = password
        else:
            raise Exception("No credentials available")

    def _setup_dataset(self):
        """Set up dataset ID"""
        try:
            _DS = gportal.datasets()["GCOM-W/AMSR2"]["LEVEL1"]
            self.DS_L1B_TB = _DS["L1B-Brightness temperature（TB）"][0]
        except Exception as e:
            print(f"Error setting up dataset: {e}")
            raise

    def check_availability(self, date_str: str, orbit_type: str = None) -> List[Dict]:
        """
        Check data availability for a specific date

        Args:
            date_str: Date in YYYY-MM-DD format
            orbit_type: 'A' for ascending, 'D' for descending, None for all

        Returns:
            List of available files
        """
        try:
            date = dt.datetime.strptime(date_str, "%Y-%m-%d").date()

            # Search for data
            res = gportal.search(
                dataset_ids=[self.DS_L1B_TB],
                start_time=f"{date}T00:00:00",
                end_time=f"{date}T23:59:59"
            )

            print(f"{date_str}: {res.matched()} granules found")

            # Filter by orbit type if specified
            files = []
            for p in res.products():
                # Extract orbit type from identifier
                ad_flag = p["identifier"].split("_")[2][-1]

                if orbit_type is None or ad_flag == orbit_type:
                    files.append({
                        'name': p["identifier"],
                        'product': p,
                        'orbit_type': ad_flag,
                        'size': p.get('size', 'Unknown')
                    })

            return files

        except Exception as e:
            print(f"Error checking availability: {e}")
            return []

    def list_files_for_date(self, date_str: str) -> List[Dict]:
        """Get all files for a specific date"""
        return self.check_availability(date_str, orbit_type=None)

    def download_files(self, date_str: str, orbit_type: str, output_dir: pathlib.Path,
                       progress_callback=None) -> List[pathlib.Path]:
        """
        Download all files for a specific date and orbit type

        Args:
            date_str: Date in YYYY-MM-DD format
            orbit_type: 'A' for ascending, 'D' for descending
            output_dir: Directory to save files
            progress_callback: Function to call with progress updates

        Returns:
            List of downloaded file paths
        """
        try:
            # Get available files
            files = self.check_availability(date_str, orbit_type)

            if not files:
                return []

            downloaded_files = []

            # Download each file
            for i, file_info in enumerate(files):
                if progress_callback:
                    progress_callback(f"Downloading file {i + 1}/{len(files)}: {file_info['name']}")

                try:
                    local_path = gportal.download(
                        file_info['product'],
                        local_dir=str(output_dir)
                    )
                    downloaded_files.append(pathlib.Path(local_path))

                except Exception as e:
                    print(f"Error downloading {file_info['name']}: {e}")
                    continue

            return downloaded_files

        except Exception as e:
            print(f"Error in download_files: {e}")
            return []

    def download_single_file(self, file_info: Dict, output_dir: pathlib.Path) -> Optional[pathlib.Path]:
        """
        Download a single file

        Args:
            file_info: File information dictionary
            output_dir: Directory to save file

        Returns:
            Path to downloaded file or None
        """
        try:
            local_path = gportal.download(
                file_info['product'],
                local_dir=str(output_dir)
            )
            return pathlib.Path(local_path)

        except Exception as e:
            print(f"Error downloading file: {e}")
            return None

    def fetch_amsr2_organized(self, date_str: str, base_dir: pathlib.Path):
        """
        Download and organize AMSR-2 data by orbit type
        Based on user's original code
        """
        date = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
        day_dir = base_dir / date.strftime("%Y%m%d")
        asc_dir = day_dir / "Ascending"
        des_dir = day_dir / "Descending"
        asc_dir.mkdir(parents=True, exist_ok=True)
        des_dir.mkdir(parents=True, exist_ok=True)

        # Search for all files in the day
        res = gportal.search(
            dataset_ids=[self.DS_L1B_TB],
            start_time=f"{date}T00:00:00",
            end_time=f"{date}T23:59:59"
        )

        print(f"{date_str}: {res.matched()} granules")

        # Separate by orbit type
        asc_prod, des_prod = [], []
        for p in res.products():
            ad_flag = p["identifier"].split("_")[2][-1]
            (asc_prod if ad_flag == "A" else des_prod).append(p)

        # Download to respective directories
        if asc_prod:
            self._download_batch(asc_prod, asc_dir)
        if des_prod:
            self._download_batch(des_prod, des_dir)

    def _download_batch(self, products, out_dir):
        """Download a batch of products with progress bar"""
        for p in tqdm.tqdm(products, desc=f"→ {out_dir.name}", leave=False):
            try:
                gportal.download(p, local_dir=str(out_dir))
            except Exception as e:
                print(f"Error downloading {p['identifier']}: {e}")
                continue