"""
Main CLI interface for SatelliteProcessor
"""

import os
import sys
import pathlib
import getpass
from datetime import datetime
from typing import Optional, List, Dict

from core.gportal_client import GPortalClient
from core.image_processor import ImageProcessor
from core.data_handler import DataHandler
from core.enhanced_processor import EnhancedProcessor
from utils.validators import DateValidator
from .menu_handler import MenuHandler
from .progress_display import ProgressDisplay


class SatelliteProcessorCLI:
    """Command Line Interface for SatelliteProcessor"""

    def __init__(self, auth_manager, path_manager, file_manager):
        self.auth_manager = auth_manager
        self.path_manager = path_manager
        self.file_manager = file_manager

        # Initialize components
        self.menu_handler = MenuHandler()
        self.progress = ProgressDisplay()
        self.date_validator = DateValidator()

        # Will be initialized after authentication
        self.gportal_client = None
        self.image_processor = ImageProcessor()
        self.data_handler = DataHandler()
        self.enhanced_processor = None

        # State
        self.authenticated = False
        self.output_path_set = False

    def run(self):
        """Main CLI loop"""
        self.progress.show_banner()

        # Step 1: Authentication
        if not self._handle_authentication():
            print("\nAuthentication failed. Exiting...")
            return

        # Step 2: Output path setup
        if not self._handle_output_path():
            print("\nOutput path setup failed. Exiting...")
            return

        # Step 3: Main menu loop
        self._main_menu_loop()

        print("\nThank you for using SatelliteProcessor!")

    def _handle_authentication(self) -> bool:
        """Handle authentication process"""
        # Check existing credentials
        if self.auth_manager.has_credentials():
            username, password = self.auth_manager.get_credentials()
            print(f"\nFound saved credentials for user: {username}")

            use_saved = input("Use saved credentials? (y/n): ").strip().lower()
            if use_saved == 'y':
                print("Testing credentials...")
                if self.auth_manager.test_credentials(username, password):
                    self.progress.success("Authentication successful!")
                    self.authenticated = True
                    self._initialize_clients()
                    return True
                else:
                    self.progress.error("Saved credentials are invalid")

        # Get new credentials
        print("\n=== GPORTAL Authentication ===")
        for attempt in range(3):
            username = input("Username: ").strip()
            password = getpass.getpass("Password: ")

            print("\nTesting credentials...")
            if self.auth_manager.test_credentials(username, password):
                self.auth_manager.save_credentials(username, password)
                self.progress.success("Authentication successful!")
                self.authenticated = True
                self._initialize_clients()
                return True
            else:
                remaining = 2 - attempt
                if remaining > 0:
                    self.progress.error(f"Invalid credentials. {remaining} attempts remaining.")
                else:
                    self.progress.error("Authentication failed after 3 attempts.")

        return False

    def _handle_output_path(self) -> bool:
        """Handle output path setup"""
        # Check existing path
        if self.path_manager.has_output_path():
            output_path = self.path_manager.get_output_path()
            print(f"\nCurrent output directory: {output_path}")

            use_existing = input("Use this directory? (y/n): ").strip().lower()
            if use_existing == 'y':
                self.output_path_set = True
                return True

        # Get new path
        print("\n=== Output Directory Setup ===")
        while True:
            path_input = input("Enter output directory path: ").strip()

            if not path_input:
                self.progress.error("Path cannot be empty")
                continue

            try:
                output_path = pathlib.Path(path_input).expanduser().resolve()

                # Create SatData subdirectory
                sat_data_path = output_path / "SatData"
                sat_data_path.mkdir(parents=True, exist_ok=True)

                # Save path
                self.path_manager.save_output_path(sat_data_path)
                self.progress.success(f"Output directory set to: {sat_data_path}")
                self.output_path_set = True
                return True

            except Exception as e:
                self.progress.error(f"Failed to create directory: {e}")
                retry = input("Try again? (y/n): ").strip().lower()
                if retry != 'y':
                    return False

    def _initialize_clients(self):
        """Initialize API clients after authentication"""
        self.gportal_client = GPortalClient(self.auth_manager)

        # Initialize enhanced processor if model is available
        model_path = pathlib.Path(__file__).parent.parent / "ml_models" / "checkpoints" / "net_g_45738.pth"
        if model_path.exists():
            try:
                self.enhanced_processor = EnhancedProcessor(model_path)
                self.progress.info("ML enhancement model loaded successfully")
            except Exception as e:
                self.progress.warning(f"ML model could not be loaded: {e}")
                self.enhanced_processor = None
        else:
            self.progress.warning("ML enhancement model not found. Functions 3&4 will be unavailable.")

    def _main_menu_loop(self):
        """Main menu interaction loop"""
        while True:
            choice = self.menu_handler.show_main_menu()

            if choice == '1':
                self._process_polar_circle()
            elif choice == '2':
                self._process_single_strip()
            elif choice == '3':
                if self.enhanced_processor:
                    self._process_8x_enhancement()
                else:
                    self.progress.error("ML model not available for this function")
            elif choice == '4':
                if self.enhanced_processor:
                    self._process_8x_polar()
                else:
                    self.progress.error("ML model not available for this function")
            elif choice.lower() == 'exit' or choice == '5':
                break
            else:
                self.progress.error("Invalid choice. Please try again.")

            # Pause before returning to menu
            input("\nPress Enter to continue...")

    def _process_polar_circle(self):
        """Process Function 1: Polar Circle"""
        print("\n=== Function 1: Create Polar Circle ===")

        # Get date
        date_str = input("Enter date (MM/DD/YYYY): ").strip()
        is_valid, error_msg, date_obj = self.date_validator.validate_date(date_str)

        if not is_valid:
            self.progress.error(error_msg)
            return

        # Get orbit type
        orbit_type = self.menu_handler.get_orbit_type()
        if not orbit_type:
            return

        # Get pole
        pole = self.menu_handler.get_pole()
        if not pole:
            return

        # Process
        self.progress.start_spinner("Checking data availability...")

        try:
            # Convert date for API
            api_date = date_obj.strftime("%Y-%m-%d")

            # Check availability
            available_files = self.gportal_client.check_availability(api_date, orbit_type)
            self.progress.stop_spinner()

            if not available_files:
                self.progress.error(f"No data available for {date_str}")
                return

            print(f"\nFound {len(available_files)} files")

            # Confirm processing
            confirm = input(f"Process {len(available_files)} files? (y/n): ").strip().lower()
            if confirm != 'y':
                print("Cancelled.")
                return

            # Download files
            print("\nDownloading files...")
            temp_dir = self.file_manager.get_temp_dir()

            downloaded_files = []
            with self.progress.create_progress_bar(len(available_files), "Downloading") as pbar:
                for i, file_info in enumerate(available_files):
                    pbar.set_description(f"Downloading {file_info['name'][:30]}...")

                    local_file = self.gportal_client.download_single_file(file_info, temp_dir)
                    if local_file:
                        downloaded_files.append(local_file)

                    pbar.update(1)

            if not downloaded_files:
                self.progress.error("Failed to download files")
                return

            # Process polar image
            print("\nCreating polar image...")
            self.progress.start_spinner("Processing...")

            result_data = self.image_processor.create_polar_image(
                downloaded_files, orbit_type, pole
            )

            self.progress.stop_spinner()

            if result_data is None:
                self.progress.error("Failed to create polar image")
                return

            # Save results
            output_base = self.path_manager.get_output_path()
            output_dir = output_base / f"{api_date}-{orbit_type}-{pole}"
            output_dir.mkdir(parents=True, exist_ok=True)

            print("Saving results...")

            # Save images
            self.image_processor.save_color_image(result_data, output_dir / "polar_color.png")
            self.image_processor.save_grayscale_image(result_data, output_dir / "polar_grayscale.png")
            self.data_handler.save_temperature_array(result_data, output_dir / "temperature_data.npz")

            # Cleanup
            self.file_manager.cleanup_temp()

            self.progress.success(f"\nProcessing complete! Results saved to:\n{output_dir}")

        except Exception as e:
            self.progress.stop_spinner()
            self.progress.error(f"Processing failed: {e}")

    def _process_single_strip(self):
        """Process Function 2: Single Strip"""
        print("\n=== Function 2: Process Single Strip ===")

        # Get date
        date_str = input("Enter date (MM/DD/YYYY): ").strip()
        is_valid, error_msg, date_obj = self.date_validator.validate_date(date_str)

        if not is_valid:
            self.progress.error(error_msg)
            return

        # Check available files
        self.progress.start_spinner("Checking available files...")

        try:
            api_date = date_obj.strftime("%Y-%m-%d")
            all_files = self.gportal_client.list_files_for_date(api_date)
            self.progress.stop_spinner()

            if not all_files:
                self.progress.error(f"No files available for {date_str}")
                return

            # Display files
            print(f"\nFound {len(all_files)} files:")
            print("-" * 60)
            for i, file_info in enumerate(all_files):
                print(f"{i + 1:3d}. {file_info['name']}")
            print("-" * 60)

            # Get selection
            while True:
                try:
                    selection = input("\nSelect file number (1-{}): ".format(len(all_files))).strip()
                    file_idx = int(selection) - 1

                    if 0 <= file_idx < len(all_files):
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(all_files)}")
                except ValueError:
                    print("Please enter a valid number")

            selected_file = all_files[file_idx]
            print(f"\nSelected: {selected_file['name']}")

            # Download and process
            print("\nDownloading file...")
            temp_dir = self.file_manager.get_temp_dir()

            self.progress.start_spinner("Downloading...")
            downloaded_file = self.gportal_client.download_single_file(selected_file, temp_dir)
            self.progress.stop_spinner()

            if not downloaded_file:
                self.progress.error("Failed to download file")
                return

            # Extract temperature data
            print("Processing data...")
            self.progress.start_spinner("Extracting temperature data...")

            temp_data, scale_factor = self.data_handler.extract_temperature_data(downloaded_file)
            self.progress.stop_spinner()

            if temp_data is None:
                self.progress.error("Failed to extract temperature data")
                return

            # Save results
            date_str_formatted = date_str.replace("/", "-")
            output_base = self.path_manager.get_output_path()
            output_dir = output_base / f"SingleStrip-{date_str_formatted}"
            output_dir.mkdir(parents=True, exist_ok=True)

            print("Saving results...")

            # Save images
            base_name = selected_file['name'].replace('.h5', '')
            self.image_processor.save_color_image(temp_data, output_dir / f"{base_name}_color.png")
            self.image_processor.save_grayscale_image(temp_data, output_dir / f"{base_name}_grayscale.png")
            self.data_handler.save_temperature_array(temp_data, output_dir / f"{base_name}_temperature.npz")

            # Cleanup
            self.file_manager.cleanup_temp()

            self.progress.success(f"\nProcessing complete! Results saved to:\n{output_dir}")

        except Exception as e:
            self.progress.stop_spinner()
            self.progress.error(f"Processing failed: {e}")

    def _process_8x_enhancement(self):
        """Process Function 3: 8x Enhancement"""
        print("\n=== Function 3: 8x Quality Enhancement ===")

        # Similar to single strip but with enhancement
        date_str = input("Enter date (MM/DD/YYYY): ").strip()
        is_valid, error_msg, date_obj = self.date_validator.validate_date(date_str)

        if not is_valid:
            self.progress.error(error_msg)
            return

        # Check available files
        self.progress.start_spinner("Checking available files...")

        try:
            api_date = date_obj.strftime("%Y-%m-%d")
            all_files = self.gportal_client.list_files_for_date(api_date)
            self.progress.stop_spinner()

            if not all_files:
                self.progress.error(f"No files available for {date_str}")
                return

            # Display files
            print(f"\nFound {len(all_files)} files:")
            print("-" * 60)
            for i, file_info in enumerate(all_files):
                print(f"{i + 1:3d}. {file_info['name']}")
            print("-" * 60)

            # Get selection
            while True:
                try:
                    selection = input("\nSelect file number (1-{}): ".format(len(all_files))).strip()
                    file_idx = int(selection) - 1

                    if 0 <= file_idx < len(all_files):
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(all_files)}")
                except ValueError:
                    print("Please enter a valid number")

            selected_file = all_files[file_idx]
            print(f"\nSelected: {selected_file['name']}")

            # Download file
            temp_dir = self.file_manager.get_temp_dir()
            self.progress.start_spinner("Downloading...")
            downloaded_file = self.gportal_client.download_single_file(selected_file, temp_dir)
            self.progress.stop_spinner()

            if not downloaded_file:
                self.progress.error("Failed to download file")
                return

            # Extract data
            print("Extracting temperature data...")
            temp_data, scale_factor = self.data_handler.extract_temperature_data(downloaded_file)

            if temp_data is None:
                self.progress.error("Failed to extract temperature data")
                return

            # Extract coordinates
            print("Extracting coordinates...")
            lat, lon = self.enhanced_processor.extract_coordinates_from_h5(downloaded_file)

            # Apply 8x enhancement
            print("\nApplying 8x enhancement (this may take several minutes)...")
            print("Stage 1: First 2x enhancement...")
            print("Stage 2: Second 2x enhancement (4x total)...")
            print("Stage 3: Third 2x enhancement (8x total)...")

            metadata = {
                'filename': selected_file['name'],
                'orbit_type': selected_file.get('orbit_type', 'unknown'),
                'scale_factor': scale_factor
            }

            # Process with progress tracking
            self.progress.start_spinner("Processing with ML model...")
            enhanced_results = self.enhanced_processor.sr_processor.process_single_strip_8x(
                temp_data, lat, lon, metadata
            )
            self.progress.stop_spinner()

            # Save results
            date_str_formatted = date_str.replace("/", "-")
            output_base = self.path_manager.get_output_path()
            output_dir = output_base / f"Enhanced8x-{date_str_formatted}"

            print("Saving enhanced results...")
            self.enhanced_processor.save_enhanced_results(
                enhanced_results,
                output_dir,
                selected_file['name'].replace('.h5', ''),
                percentile_filter=True
            )

            # Cleanup
            self.file_manager.cleanup_temp()

            self.progress.success(f"\n8x Enhancement complete! Results saved to:\n{output_dir}")

        except Exception as e:
            self.progress.stop_spinner()
            self.progress.error(f"Enhancement failed: {e}")

    def _process_8x_polar(self):
        """Process Function 4: 8x Enhanced Polar Circle"""
        print("\n=== Function 4: 8x Enhanced Polar Circle ===")

        # Get inputs (similar to polar circle)
        date_str = input("Enter date (MM/DD/YYYY): ").strip()
        is_valid, error_msg, date_obj = self.date_validator.validate_date(date_str)

        if not is_valid:
            self.progress.error(error_msg)
            return

        orbit_type = self.menu_handler.get_orbit_type()
        if not orbit_type:
            return

        pole = self.menu_handler.get_pole()
        if not pole:
            return

        # Process
        self.progress.start_spinner("Checking data availability...")

        try:
            api_date = date_obj.strftime("%Y-%m-%d")
            available_files = self.gportal_client.check_availability(api_date, orbit_type)
            self.progress.stop_spinner()

            if not available_files:
                self.progress.error(f"No data available for {date_str}")
                return

            print(f"\nFound {len(available_files)} files")
            confirm = input(
                f"Process and enhance {len(available_files)} files? This may take a long time. (y/n): ").strip().lower()
            if confirm != 'y':
                print("Cancelled.")
                return

            # Download files
            print("\nDownloading files...")
            temp_dir = self.file_manager.get_temp_dir()

            downloaded_files = []
            with self.progress.create_progress_bar(len(available_files), "Downloading") as pbar:
                for i, file_info in enumerate(available_files):
                    pbar.set_description(f"Downloading {file_info['name'][:30]}...")

                    local_file = self.gportal_client.download_single_file(file_info, temp_dir)
                    if local_file:
                        downloaded_files.append(local_file)

                    pbar.update(1)

            # Process with 8x enhancement
            print("\nApplying 8x enhancement to polar projection...")
            print("This process will:")
            print("1. Enhance each swath to 8x resolution")
            print("2. Create enhanced polar projection")
            print("3. Apply intelligent hole filling")
            print("\nThis may take 10-30 minutes depending on the number of files...")

            self.progress.start_spinner("Processing with ML enhancement...")
            enhanced_result = self.enhanced_processor.sr_processor.process_polar_8x_enhanced(
                downloaded_files, orbit_type, pole
            )
            self.progress.stop_spinner()

            # Save results
            output_base = self.path_manager.get_output_path()
            output_dir = output_base / f"{api_date}-{orbit_type}-{pole}-Enhanced8x"
            output_dir.mkdir(parents=True, exist_ok=True)

            print("Saving enhanced polar image...")

            # Get enhanced data
            polar_temp_8x = enhanced_result['temperature_8x']

            # Save images
            self.image_processor.save_color_image_percentile(polar_temp_8x, output_dir / "polar_enhanced_8x_color.png")
            self.image_processor.save_grayscale_image_percentile(polar_temp_8x,
                                                                 output_dir / "polar_enhanced_8x_grayscale.png")

            # Save data
            import numpy as np
            np.savez_compressed(
                output_dir / "temperature_data_enhanced_8x.npz",
                temperature=polar_temp_8x,
                statistics=enhanced_result['statistics'],
                metadata=enhanced_result['metadata']
            )

            # Cleanup
            self.file_manager.cleanup_temp()

            self.progress.success(f"\n8x Enhanced processing complete! Results saved to:\n{output_dir}")

            # Print statistics
            stats = enhanced_result['statistics']
            print(f"\nEnhancement Statistics:")
            print(f"  Grid size: {polar_temp_8x.shape} (8x larger than standard)")
            print(f"  Coverage: {stats['coverage_percent']:.1f}%")
            print(f"  Temperature range: [{stats['min_temp']:.1f}, {stats['max_temp']:.1f}] K")

        except Exception as e:
            self.progress.stop_spinner()
            self.progress.error(f"Processing failed: {e}")

    # Batch mode methods
    def initialize_batch_mode(self) -> bool:
        """Initialize for batch mode operation"""
        # Check credentials
        if not self.auth_manager.has_credentials():
            print("ERROR: No saved credentials found. Run in interactive mode first.")
            return False

        username, password = self.auth_manager.get_credentials()
        if not self.auth_manager.test_credentials(username, password):
            print("ERROR: Saved credentials are invalid.")
            return False

        # Check output path
        if not self.path_manager.has_output_path():
            print("ERROR: No output path configured. Run in interactive mode first.")
            return False

        # Initialize clients
        self._initialize_clients()
        return True

    def process_polar_circle_batch(self, date_str: str, orbit_type: str, pole: str):
        """Batch mode polar circle processing"""
        print(f"Processing polar circle: date={date_str}, orbit={orbit_type}, pole={pole}")

        # Validate date
        is_valid, error_msg, date_obj = self.date_validator.validate_date(date_str)
        if not is_valid:
            raise ValueError(f"Invalid date: {error_msg}")

        # Rest of the processing logic...
        # (Similar to interactive but without prompts)

    def process_single_strip_batch(self, date_str: str, file_index: int):
        """Batch mode single strip processing"""
        print(f"Processing single strip: date={date_str}, file_index={file_index}")
        # Implementation...

    def process_enhancement_batch(self, date_str: str, file_index: int):
        """Batch mode 8x enhancement"""
        if not self.enhanced_processor:
            raise RuntimeError("ML model not available")
        print(f"Processing 8x enhancement: date={date_str}, file_index={file_index}")
        # Implementation...

    def process_enhanced_polar_batch(self, date_str: str, orbit_type: str, pole: str):
        """Batch mode 8x enhanced polar"""
        if not self.enhanced_processor:
            raise RuntimeError("ML model not available")
        print(f"Processing 8x enhanced polar: date={date_str}, orbit={orbit_type}, pole={pole}")
        # Implementation...