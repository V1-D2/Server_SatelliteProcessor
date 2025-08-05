#!/usr/bin/env python3
"""
Server-side single strip processing
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


def process_single_strip(params: dict, output_dir: pathlib.Path, job_id: str) -> bool:
    """
    Process single strip data

    Args:
        params: Job parameters (date, file_index, credentials)
        output_dir: Directory for results
        job_id: Unique job identifier

    Returns:
        True if successful, False otherwise
    """
    try:
        # Extract parameters
        date_str = params['date']
        file_index = params.get('file_index', 0)
        file_name = params.get('file_name', None)
        credentials = params.get('credentials', {})

        logger.info(f"Processing single strip: date={date_str}, file_index={file_index}")

        # Setup managers
        auth_manager = AuthManager()
        if credentials:
            auth_manager.save_credentials(credentials['username'], credentials['password'])

        file_manager = FileManager()

        # Initialize processors
        gportal_client = GPortalClient(auth_manager)
        image_processor = ImageProcessor()
        data_handler = DataHandler()

        # Create temp directory
        temp_dir = pathlib.Path(f"/tmp/satproc_{job_id}")
        temp_dir.mkdir(exist_ok=True)

        try:
            # Get available files for date
            all_files = gportal_client.list_files_for_date(date_str)

            if not all_files:
                logger.error(f"No files available for {date_str}")
                return False

            # Select file
            if file_name:
                # Find by name
                file_info = None
                for f in all_files:
                    if f['name'] == file_name:
                        file_info = f
                        break
                if not file_info:
                    logger.error(f"File {file_name} not found")
                    return False
            else:
                # Use index
                if file_index >= len(all_files):
                    logger.error(f"File index {file_index} out of range")
                    return False
                file_info = all_files[file_index]

            logger.info(f"Processing file: {file_info['name']}")

            # Download file
            downloaded_file = gportal_client.download_single_file(file_info, temp_dir)

            if not downloaded_file:
                logger.error("Failed to download file")
                return False

            # Extract temperature data
            logger.info("Extracting temperature data...")
            temp_data, scale_factor = data_handler.extract_temperature_data(downloaded_file)

            if temp_data is None:
                logger.error("Failed to extract temperature data")
                return False

            # Save outputs
            logger.info("Saving results...")

            # Base filename without extension
            base_name = file_info['name'].replace('.h5', '')

            # Save color image
            color_path = output_dir / f"{base_name}_color.png"
            image_processor.save_color_image(temp_data, color_path)

            # Save color image with percentile filter
            color_percentile_path = output_dir / f"{base_name}_color_percentile.png"
            image_processor.save_color_image_percentile(temp_data, color_percentile_path)

            # Save viridis image
            viridis_path = output_dir / f"{base_name}_viridis.png"
            image_processor.save_viridis_image(temp_data, viridis_path)

            # Save grayscale image
            gray_path = output_dir / f"{base_name}_grayscale.png"
            image_processor.save_grayscale_image(temp_data, gray_path)

            # Save grayscale with percentile filter
            gray_percentile_path = output_dir / f"{base_name}_grayscale_percentile.png"
            image_processor.save_grayscale_image_percentile(temp_data, gray_percentile_path)

            # Save temperature data
            temp_path = output_dir / f"{base_name}_temperature.npz"
            data_handler.save_temperature_array(temp_data, temp_path)

            # Save metadata
            metadata = {
                'job_id': job_id,
                'function': 'single_strip',
                'parameters': params,
                'date_processed': datetime.now().isoformat(),
                'file_processed': file_info['name'],
                'scale_factor': float(scale_factor),
                'output_files': [
                    f"{base_name}_color.png",
                    f"{base_name}_color_percentile.png",
                    f"{base_name}_viridis.png",
                    f"{base_name}_grayscale.png",
                    f"{base_name}_grayscale_percentile.png",
                    f"{base_name}_temperature.npz"
                ]
            }

            with open(output_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Single strip processing completed successfully")
            return True

        finally:
            # Cleanup temp files
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    except Exception as e:
        logger.error(f"Error in single strip processing: {e}")
        return False


if __name__ == "__main__":
    # Test mode
    import sys

    if len(sys.argv) > 1:
        params = {
            'date': sys.argv[1] if len(sys.argv) > 1 else '2025-05-26',
            'file_index': int(sys.argv[2]) if len(sys.argv) > 2 else 0
        }

        output_dir = pathlib.Path('./test_output')
        output_dir.mkdir(exist_ok=True)

        success = process_single_strip(params, output_dir, 'test_job')
        print(f"Success: {success}")