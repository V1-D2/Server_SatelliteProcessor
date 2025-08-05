#!/usr/bin/env python3
"""
Main job processor for Server_SatelliteProcessor
Monitors job queue and dispatches processing tasks
"""

import os
import sys
import json
import time
import shutil
import pathlib
import logging
import subprocess
from datetime import datetime
from typing import Dict, Optional

# Setup paths
SERVER_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(SERVER_ROOT))

# Import processing modules
from scripts.polar_circle import process_polar_circle
from scripts.single_strip import process_single_strip
from scripts.enhance_8x import process_enhance_8x
from scripts.polar_enhanced_8x import process_polar_enhanced_8x

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(SERVER_ROOT / 'logs' / 'job_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class JobProcessor:
    """Main job processing engine"""

    def __init__(self):
        self.server_root = SERVER_ROOT
        self.jobs_dir = self.server_root / 'jobs'
        self.results_dir = self.server_root / 'results'
        self.logs_dir = self.server_root / 'logs'

        # Ensure directories exist
        for dir_path in [self.jobs_dir / 'pending',
                         self.jobs_dir / 'running',
                         self.jobs_dir / 'completed',
                         self.jobs_dir / 'failed',
                         self.results_dir,
                         self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def run(self):
        """Main processing loop"""
        logger.info("Job processor started")

        while True:
            try:
                # Check for new jobs
                pending_jobs = list((self.jobs_dir / 'pending').glob('*.json'))

                if pending_jobs:
                    # Process oldest job first
                    job_file = min(pending_jobs, key=lambda p: p.stat().st_mtime)
                    self.process_job(job_file)
                else:
                    # No jobs, wait
                    time.sleep(5)

            except KeyboardInterrupt:
                logger.info("Job processor stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(10)

    def process_job(self, job_file: pathlib.Path):
        """Process a single job"""
        job_id = job_file.stem
        logger.info(f"Processing job: {job_id}")

        try:
            # Load job data
            with open(job_file, 'r') as f:
                job_data = json.load(f)

            # Move to running
            running_file = self.jobs_dir / 'running' / job_file.name
            shutil.move(str(job_file), str(running_file))

            # Update status
            job_data['status'] = 'running'
            job_data['start_time'] = datetime.now().isoformat()
            self._save_job_status(running_file, job_data)

            # Create job-specific result directory
            job_result_dir = self.results_dir / job_id
            job_result_dir.mkdir(exist_ok=True)

            # Process based on function type
            function = job_data['function']
            params = job_data['parameters']

            if function == 'polar_circle':
                success = process_polar_circle(params, job_result_dir, job_id)
            elif function == 'single_strip':
                success = process_single_strip(params, job_result_dir, job_id)
            elif function == 'enhance_8x':
                success = process_enhance_8x(params, job_result_dir, job_id)
            elif function == 'polar_enhanced_8x':
                success = process_polar_enhanced_8x(params, job_result_dir, job_id)
            else:
                raise ValueError(f"Unknown function: {function}")

            # Move to completed or failed
            if success:
                completed_file = self.jobs_dir / 'completed' / job_file.name
                shutil.move(str(running_file), str(completed_file))
                job_data['status'] = 'completed'
                job_data['result_path'] = str(job_result_dir)
                logger.info(f"Job {job_id} completed successfully")
            else:
                failed_file = self.jobs_dir / 'failed' / job_file.name
                shutil.move(str(running_file), str(failed_file))
                job_data['status'] = 'failed'
                logger.error(f"Job {job_id} failed")

            # Update final status
            job_data['end_time'] = datetime.now().isoformat()
            final_file = completed_file if success else failed_file
            self._save_job_status(final_file, job_data)

        except Exception as e:
            logger.error(f"Error processing job {job_id}: {e}")
            # Move to failed
            try:
                failed_file = self.jobs_dir / 'failed' / job_file.name
                if running_file.exists():
                    shutil.move(str(running_file), str(failed_file))
                elif job_file.exists():
                    shutil.move(str(job_file), str(failed_file))
            except:
                pass

    def _save_job_status(self, job_file: pathlib.Path, job_data: Dict):
        """Save job status"""
        with open(job_file, 'w') as f:
            json.dump(job_data, f, indent=2)


if __name__ == "__main__":
    processor = JobProcessor()
    processor.run()