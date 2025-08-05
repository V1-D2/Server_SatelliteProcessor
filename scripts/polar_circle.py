#!/usr/bin/env python3
"""
Server-side polar circle processing
"""

import sys
import pathlib
import logging
import json
import shutil
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from core.gportal_client import GPortalClient
from core.image_processor import ImageProcessor
from core.data_handler import DataHandler
from core.auth_manager import AuthManager
from utils.file_manager import FileManager

logger = logging.getLogger(__name__)


def process_polar_circle(params: dict, output_dir: pathlib.Path, job_id: str) -> bool:
    """
    Process polar circle creation

    Args:
        params: Job parameters (date, orbit_type, pole, credentials)
        output_dir: Directory for results
        job_id: Unique job identifier

    Returns:
        True if successful, False otherwise
    """
    try:
        # Extract parameters
        date_str = params['date']
        orbit_type = params['orbit_type']
        pole = params['pole']
        credentials = params.get('credentials', {})

        logger.info(f"Processing polar circle: date={date_str}, orbit={orbit_type}, pole={pole}")

        # Setup managers
        auth_manager = AuthManager()
        if credentials:
            auth_manager.save_credentials(credentials['username'], credentials['password'])

        file_manager = FileManager()

        # Initialize processors
        gportal_client = GPortalClient(auth_manager)
        image_processor = ImageProcessor()
        data_handler = DataHandler()

        # Create temp directory for downloads
        temp_dir = pathlib.Path(f"/tmp/satproc_{job_id}")
        temp_dir.mkdir(exist_ok=True)

        try:
            # Check data availability
            available_files = gportal_client.check_availability(date_str, orbit_type)

            if not available_files:
                logger.error(f"No data available for {date_str}")
                return False

            logger.info(f"Found {len(available_files)} files to process")

            # Download files
            downloaded_files = gportal_client.download_files(
                date_str, orbit_type, temp_dir
            )

            if not downloaded_files:
                logger.error("Failed to download files")
                return False

            # Process to create polar image
            logger.info("Creating polar image...")
            result_data = image_processor.create_polar_image(
                downloaded_files, orbit_type, pole
            )

            if result_data is None:
                logger.error("Failed to create polar image")
                return False

            # Save outputs
            logger.info("Saving results...")

            # Save color image
            color_path = output_dir / "polar_color.png"
            image_processor.save_color_image(result_data, color_path)

            # Save color image with percentile filter
            color_percentile_path = output_dir / "polar_color_percentile.png"
            image_processor.save_color_image_percentile(result_data, color_percentile_path)

            # Save viridis image
            viridis_path = output_dir / "polar_viridis.png"
            image_processor.save_viridis_image(result_data, viridis_path)

            # Save grayscale image
            gray_path = output_dir / "polar_grayscale.png"
            image_processor.save_grayscale_image(result_data, gray_path)

            # Save grayscale with percentile filter
            gray_percentile_path = output_dir / "polar_grayscale_percentile.png"
            image_processor.save_grayscale_image_percentile(result_data, gray_percentile_path)

            # Save temperature data
            temp_path = output_dir / "temperature_data.npz"
            data_handler.save_temperature_array(result_data, temp_path)

            # Save metadata
            metadata = {
                'job_id': job_id,
                'function': 'polar_circle',
                'parameters': params,
                'date_processed': datetime.now().isoformat(),
                'files_processed': len(downloaded_files),
                'output_files': [
                    'polar_color.png',
                    'polar_color_percentile.png',
                    'polar_viridis.png',
                    'polar_grayscale.png',
                    'polar_grayscale_percentile.png',
                    'temperature_data.npz'
                ]
            }

            with open(output_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Polar circle processing completed successfully")
            return True

        finally:
            # Cleanup temp files
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    except Exception as e:
        logger.error(f"Error in polar circle processing: {e}")
        return False


if __name__ == "__main__":
    # Test mode
    import sys

    if len(sys.argv) > 1:
        params = {
            'date': sys.argv[1] if len(sys.argv) > 1 else '2025-05-26',
            'orbit_type': sys.argv[2] if len(sys.argv) > 2 else 'A',
            'pole': sys.argv[3] if len(sys.argv) > 3 else 'N'
        }

        output_dir = pathlib.Path('./test_output')
        output_dir.mkdir(exist_ok=True)

        success = process_polar_circle(params, output_dir, 'test_job')
        print(f"Success: {success}")