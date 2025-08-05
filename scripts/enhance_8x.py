#!/usr/bin/env python3
"""
Server-side 8x enhancement processing for single files
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
from core.enhanced_processor import EnhancedProcessor
from core.data_handler import DataHandler
from core.auth_manager import AuthManager
from utils.file_manager import FileManager
from utils.device_utils import get_best_device

logger = logging.getLogger(__name__)


def process_enhance_8x(params: dict, output_dir: pathlib.Path, job_id: str) -> bool:
    """
    Process 8x enhancement for single strip

    Args:
        params: Job parameters (date, file_index/file_name, credentials)
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

        logger.info(f"Processing 8x enhancement: date={date_str}, file_index={file_index}")

        # Check for ML model
        model_path = pathlib.Path(__file__).parent.parent / "ml_models" / "checkpoints" / "net_g_45738.pth"
        if not model_path.exists():
            logger.error(f"ML model not found at {model_path}")
            return False

        # Setup managers
        auth_manager = AuthManager()
        if credentials:
            auth_manager.save_credentials(credentials['username'], credentials['password'])

        file_manager = FileManager()

        # Initialize processors
        gportal_client = GPortalClient(auth_manager)
        data_handler = DataHandler()

        # Get best device (should be GPU on server)
        device, device_name = get_best_device()
        logger.info(f"Using device: {device_name}")

        enhanced_processor = EnhancedProcessor(model_path, device=str(device))

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

            # Extract coordinates
            logger.info("Extracting coordinates...")
            lat, lon = enhanced_processor.extract_coordinates_from_h5(downloaded_file)

            # Process with 8x enhancement
            logger.info("Applying 8x enhancement (this may take a few minutes)...")

            metadata = {
                'filename': file_info['name'],
                'orbit_type': file_info.get('orbit_type', 'unknown'),
                'scale_factor': scale_factor
            }

            # Run 8x enhancement
            enhanced_results = enhanced_processor.sr_processor.process_single_strip_8x(
                temp_data, lat, lon, metadata
            )

            # Save results
            logger.info("Saving enhanced results...")

            base_name = file_info['name'].replace('.h5', '')

            enhanced_processor.save_enhanced_results(
                enhanced_results,
                output_dir,
                base_name,
                percentile_filter=True
            )

            # Save additional metadata
            job_metadata = {
                'job_id': job_id,
                'function': 'enhance_8x',
                'parameters': params,
                'date_processed': datetime.now().isoformat(),
                'file_processed': file_info['name'],
                'device_used': device_name,
                'enhancement_stats': enhanced_results['statistics'],
                'output_files': [
                    f"{base_name}_enhanced_8x.npz",
                    f"{base_name}_enhanced_8x_color.png",
                    f"{base_name}_enhanced_8x_gray.png"
                ]
            }

            with open(output_dir / 'job_metadata.json', 'w') as f:
                json.dump(job_metadata, f, indent=2)

            logger.info(f"8x enhancement completed successfully")
            return True

        finally:
            # Cleanup temp files
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

            # Clear GPU memory
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Error in 8x enhancement: {e}")
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

        success = process_enhance_8x(params, output_dir, 'test_job')
        print(f"Success: {success}")