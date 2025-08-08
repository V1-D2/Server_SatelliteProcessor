#!/usr/bin/env python3
"""
Server-side 8x enhanced polar circle processing
"""

import sys
import pathlib
import logging
import json
import shutil
from datetime import datetime
import gc
import torch

# Add parent directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from core.gportal_client import GPortalClient
from core.enhanced_processor import EnhancedProcessor
from core.image_processor import ImageProcessor
from core.auth_manager import AuthManager
from utils.file_manager import FileManager
from utils.device_utils import get_best_device
import numpy as np

logger = logging.getLogger(__name__)


def process_polar_enhanced_8x(params: dict, output_dir: pathlib.Path, job_id: str) -> bool:
    """
    Process 8x enhanced polar circle

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

        logger.info(f"Processing 8x enhanced polar: date={date_str}, orbit={orbit_type}, pole={pole}")

        # Check for ML model
        model_path = pathlib.Path(__file__).parent.parent / "ml_models" / "checkpoints" / "SwinIR-RealESRGAN_net_g_60000_4th_epoch_Strong_Discriminator.pth"
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
        image_processor = ImageProcessor()

        # Get best device (should be GPU on server)
        device, device_name = get_best_device()
        logger.info(f"Using device: {device_name}")

        enhanced_processor = EnhancedProcessor(model_path, device=str(device))

        # Create temp directory
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

            # Process with 8x enhancement
            logger.info("Applying 8x enhancement to polar projection (this may take several minutes)...")

            # Use the ML processor to create enhanced polar image
            enhanced_result = enhanced_processor.sr_processor.process_polar_8x_enhanced(
                downloaded_files,
                orbit_type,
                pole
            )

            # Save results
            logger.info("Saving enhanced polar image...")

            # Get enhanced temperature data
            polar_temp_8x = enhanced_result['temperature_8x']
            percentile_range = enhanced_result['percentile_range']

            # Save color image with percentile filtering
            color_path = output_dir / "polar_enhanced_8x_color.png"
            image_processor.save_color_image_percentile(polar_temp_8x, color_path)

            # Save grayscale image with percentile filtering
            gray_path = output_dir / "polar_enhanced_8x_grayscale.png"
            image_processor.save_grayscale_image_percentile(polar_temp_8x, gray_path)

            # Save temperature array
            temp_path = output_dir / "temperature_data_enhanced_8x.npz"
            np.savez_compressed(
                temp_path,
                temperature=polar_temp_8x,
                statistics=enhanced_result['statistics'],
                metadata=enhanced_result['metadata']
            )

            # Save job metadata
            job_metadata = {
                'job_id': job_id,
                'function': 'polar_enhanced_8x',
                'parameters': params,
                'date_processed': datetime.now().isoformat(),
                'files_processed': len(downloaded_files),
                'device_used': device_name,
                'enhancement_stats': enhanced_result['statistics'],
                'grid_size': polar_temp_8x.shape,
                'output_files': [
                    'polar_enhanced_8x_color.png',
                    'polar_enhanced_8x_grayscale.png',
                    'temperature_data_enhanced_8x.npz'
                ]
            }

            with open(output_dir / 'metadata.json', 'w') as f:
                json.dump(job_metadata, f, indent=2)

            logger.info(f"8x enhanced polar processing completed successfully")
            return True

        finally:
            # Cleanup temp files
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

            # Clear GPU memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Error in 8x enhanced polar processing: {e}")
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

        success = process_polar_enhanced_8x(params, output_dir, 'test_job')
        print(f"Success: {success}")