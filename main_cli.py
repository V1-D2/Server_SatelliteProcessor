#!/usr/bin/env python3
"""
SatelliteProcessor CLI - Command Line Interface Version
For server/HPC environments without GUI support
"""

import os
import sys
import pathlib
import argparse
import logging
from datetime import datetime

# Add project root to path
PROJECT_ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from cli.interface import SatelliteProcessorCLI
from core.auth_manager import AuthManager
from core.path_manager import PathManager
from utils.file_manager import FileManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories"""
    dirs = ['config', 'temp', 'logs']
    for dir_name in dirs:
        dir_path = PROJECT_ROOT / dir_name
        dir_path.mkdir(exist_ok=True)
        logger.debug(f"Directory ensured: {dir_path}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='SatelliteProcessor CLI - Process AMSR-2 satellite data'
    )

    parser.add_argument(
        '--batch',
        action='store_true',
        help='Run in batch mode (no interactive prompts)'
    )

    parser.add_argument(
        '--function',
        type=int,
        choices=[1, 2, 3, 4],
        help='Function to execute (1-4)'
    )

    parser.add_argument(
        '--date',
        type=str,
        help='Date in MM/DD/YYYY format'
    )

    parser.add_argument(
        '--orbit',
        type=str,
        choices=['A', 'D'],
        help='Orbit type (A=Ascending, D=Descending)'
    )

    parser.add_argument(
        '--file-index',
        type=int,
        help='File index for single strip processing'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Override output directory'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    return parser.parse_args()


def run_interactive_mode():
    """Run in interactive CLI mode"""
    logger.info("Starting SatelliteProcessor CLI in interactive mode")

    # Initialize managers
    auth_manager = AuthManager()
    path_manager = PathManager()
    file_manager = FileManager()

    # Create and run CLI interface
    cli = SatelliteProcessorCLI(auth_manager, path_manager, file_manager)

    try:
        cli.run()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        print("\n\nExiting SatelliteProcessor CLI...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\nError: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        file_manager.cleanup_temp()
        logger.info("Cleanup completed")


def run_batch_mode(args):
    """Run in batch mode with command line arguments"""
    logger.info("Starting SatelliteProcessor CLI in batch mode")

    # Initialize managers
    auth_manager = AuthManager()
    path_manager = PathManager()
    file_manager = FileManager()

    # Override output path if specified
    if args.output:
        path_manager.save_output_path(args.output)

    # Create CLI interface
    cli = SatelliteProcessorCLI(auth_manager, path_manager, file_manager)

    try:
        # Initialize CLI (handles auth and paths)
        if not cli.initialize_batch_mode():
            logger.error("Failed to initialize batch mode")
            sys.exit(1)

        # Execute requested function
        if args.function == 1:
            # Polar Circle
            if not args.date or not args.orbit:
                logger.error("Function 1 requires --date and --orbit")
                sys.exit(1)
            cli.process_polar_circle_batch(args.date, args.orbit, 'N')

        elif args.function == 2:
            # Single Strip
            if not args.date or args.file_index is None:
                logger.error("Function 2 requires --date and --file-index")
                sys.exit(1)
            cli.process_single_strip_batch(args.date, args.file_index)

        elif args.function == 3:
            # 8x Enhancement
            if not args.date or args.file_index is None:
                logger.error("Function 3 requires --date and --file-index")
                sys.exit(1)
            cli.process_enhancement_batch(args.date, args.file_index)

        elif args.function == 4:
            # 8x Enhanced Polar
            if not args.date or not args.orbit:
                logger.error("Function 4 requires --date and --orbit")
                sys.exit(1)
            cli.process_enhanced_polar_batch(args.date, args.orbit, 'N')

        logger.info("Batch processing completed successfully")

    except Exception as e:
        logger.error(f"Batch processing failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        file_manager.cleanup_temp()


def main():
    """Main entry point"""
    # Setup directories
    setup_directories()

    # Parse arguments
    args = parse_arguments()

    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Log startup info
    logger.info("=" * 60)
    logger.info("SatelliteProcessor CLI Started")
    logger.info(f"Version: 1.0.0-CLI")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info("=" * 60)

    # Run appropriate mode
    if args.batch and args.function:
        run_batch_mode(args)
    else:
        run_interactive_mode()

    logger.info("SatelliteProcessor CLI finished")


if __name__ == "__main__":
    main()