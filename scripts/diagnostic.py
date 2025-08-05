#!/usr/bin/env python3
"""
Diagnostic script for Server_SatelliteProcessor
Checks system configuration and identifies potential issues
"""

import os
import sys
import pathlib
import subprocess
import importlib
import json
from datetime import datetime

# Colors for terminal output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'

SERVER_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(SERVER_ROOT))


class ServerDiagnostic:
    """Run diagnostic checks on server setup"""

    def __init__(self):
        self.server_root = SERVER_ROOT
        self.issues = []
        self.warnings = []
        self.successes = []

    def run_all_checks(self):
        """Run all diagnostic checks"""
        print(f"{BLUE}{'=' * 80}")
        print("Server_SatelliteProcessor Diagnostic Tool")
        print(f"{'=' * 80}{RESET}\n")

        self.check_directory_structure()
        self.check_python_environment()
        self.check_dependencies()
        self.check_gpu_availability()
        self.check_slurm_configuration()
        self.check_file_permissions()
        self.check_gportal_credentials()
        self.check_ml_models()
        self.check_disk_space()
        self.check_network_connectivity()

        self.print_summary()

    def check_directory_structure(self):
        """Check if all required directories exist"""
        print(f"\n{BLUE}Checking directory structure...{RESET}")

        required_dirs = [
            'jobs/pending',
            'jobs/running',
            'jobs/completed',
            'jobs/failed',
            'results',
            'logs',
            'scripts',
            'sbatch',
            'core',
            'ml_models',
            'ml_models/checkpoints',
            'utils'
        ]

        for dir_path in required_dirs:
            full_path = self.server_root / dir_path
            if full_path.exists():
                self.log_success(f"✓ {dir_path}")
            else:
                self.log_issue(f"✗ Missing directory: {dir_path}")
                # Try to create it
                try:
                    full_path.mkdir(parents=True)
                    self.log_success(f"  → Created {dir_path}")
                except Exception as e:
                    self.log_issue(f"  → Failed to create: {e}")

    def check_python_environment(self):
        """Check Python version and virtual environment"""
        print(f"\n{BLUE}Checking Python environment...{RESET}")

        # Python version
        python_version = sys.version_info
        if python_version.major >= 3 and python_version.minor >= 8:
            self.log_success(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        else:
            self.log_issue(f"✗ Python {python_version.major}.{python_version.minor} (need 3.8+)")

        # Virtual environment
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            self.log_success("✓ Running in virtual environment")
        else:
            self.log_warning("⚠ Not running in virtual environment")

        # PYTHONPATH
        if str(self.server_root) in sys.path:
            self.log_success("✓ Server root in PYTHONPATH")
        else:
            self.log_warning("⚠ Server root not in PYTHONPATH")

    def check_dependencies(self):
        """Check if required Python packages are installed"""
        print(f"\n{BLUE}Checking Python dependencies...{RESET}")

        required_packages = {
            'numpy': '1.21.0',
            'matplotlib': '3.4.0',
            'h5py': '3.0.0',
            'pyproj': '3.0.0',
            'PIL': None,  # Pillow
            'scipy': '1.7.0',
            'xarray': '0.19.0',
            'tqdm': '4.60.0',
            'torch': '2.0.0',
            'cv2': None,  # opencv-python
            'paramiko': '2.11.0'
        }

        for package, min_version in required_packages.items():
            try:
                if package == 'PIL':
                    import PIL
                    version = PIL.__version__
                    package_name = 'Pillow'
                elif package == 'cv2':
                    import cv2
                    version = cv2.__version__
                    package_name = 'opencv-python'
                else:
                    mod = importlib.import_module(package)
                    version = getattr(mod, '__version__', 'unknown')
                    package_name = package

                if min_version and version != 'unknown':
                    from packaging import version as pkg_version
                    if pkg_version.parse(version) >= pkg_version.parse(min_version):
                        self.log_success(f"✓ {package_name} {version}")
                    else:
                        self.log_warning(f"⚠ {package_name} {version} (need >= {min_version})")
                else:
                    self.log_success(f"✓ {package_name} {version}")

            except ImportError:
                self.log_issue(f"✗ {package} not installed")

    def check_gpu_availability(self):
        """Check GPU availability and CUDA"""
        print(f"\n{BLUE}Checking GPU availability...{RESET}")

        try:
            import torch

            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                self.log_success(f"✓ CUDA available with {gpu_count} GPU(s)")

                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                    self.log_success(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            else:
                self.log_warning("⚠ CUDA not available - GPU processing will be slow")

        except Exception as e:
            self.log_issue(f"✗ Error checking GPU: {e}")

        # Check nvidia-smi
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True)
            if result.returncode == 0:
                self.log_success("✓ nvidia-smi available")
            else:
                self.log_warning("⚠ nvidia-smi not working")
        except:
            self.log_warning("⚠ nvidia-smi not found")

    def check_slurm_configuration(self):
        """Check SLURM availability"""
        print(f"\n{BLUE}Checking SLURM configuration...{RESET}")

        # Check if SLURM commands are available
        slurm_commands = ['sbatch', 'squeue', 'scancel', 'sinfo']

        for cmd in slurm_commands:
            try:
                result = subprocess.run([cmd, '--version'], capture_output=True)
                if result.returncode == 0:
                    self.log_success(f"✓ {cmd} available")
                else:
                    self.log_issue(f"✗ {cmd} not working")
            except:
                self.log_issue(f"✗ {cmd} not found")

        # Check partition availability
        try:
            result = subprocess.run(['sinfo', '-p', 'salvador', '-o', '%P %a'],
                                    capture_output=True, text=True)
            if result.returncode == 0 and 'salvador' in result.stdout:
                self.log_success("✓ salvador partition available")
            else:
                self.log_warning("⚠ salvador partition not found")
        except:
            pass

    def check_file_permissions(self):
        """Check file permissions"""
        print(f"\n{BLUE}Checking file permissions...{RESET}")

        # Check if scripts are executable
        script_files = [
            'scripts/job_processor.py',
            'scripts/polar_circle.py',
            'scripts/single_strip.py',
            'scripts/enhance_8x.py',
            'scripts/polar_enhanced_8x.py',
            'sbatch/process_job.sbatch'
        ]

        for script in script_files:
            script_path = self.server_root / script
            if script_path.exists():
                if os.access(script_path, os.X_OK):
                    self.log_success(f"✓ {script} is executable")
                else:
                    self.log_warning(f"⚠ {script} is not executable")
                    # Try to make it executable
                    try:
                        script_path.chmod(0o755)
                        self.log_success(f"  → Made executable")
                    except:
                        self.log_issue(f"  → Failed to make executable")
            else:
                self.log_issue(f"✗ {script} not found")

    def check_gportal_credentials(self):
        """Check GPORTAL credentials"""
        print(f"\n{BLUE}Checking GPORTAL configuration...{RESET}")

        # Check for credentials file
        cred_file = self.server_root / 'config' / 'credentials.txt'
        if cred_file.exists():
            self.log_success("✓ Credentials file exists")
            try:
                with open(cred_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) >= 2:
                        self.log_success("✓ Credentials format appears correct")
                    else:
                        self.log_warning("⚠ Credentials file may be incomplete")
            except:
                self.log_issue("✗ Cannot read credentials file")
        else:
            self.log_warning("⚠ No credentials file found (will use job-provided credentials)")

        # Check if gportal module is available
        try:
            import gportal
            self.log_success("✓ gportal module available")
        except ImportError:
            self.log_issue("✗ gportal module not installed")

    def check_ml_models(self):
        """Check ML model files"""
        print(f"\n{BLUE}Checking ML models...{RESET}")

        model_path = self.server_root / 'ml_models' / 'checkpoints' / 'net_g_45738.pth'

        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            self.log_success(f"✓ ML model found ({size_mb:.1f} MB)")

            # Try to load it
            try:
                import torch
                checkpoint = torch.load(model_path, map_location='cpu')
                self.log_success("✓ ML model loads correctly")
            except Exception as e:
                self.log_issue(f"✗ Error loading model: {e}")
        else:
            self.log_issue("✗ ML model net_g_45738.pth not found")
            self.log_issue("  Functions 3 and 4 will not work without this model")

    def check_disk_space(self):
        """Check available disk space"""
        print(f"\n{BLUE}Checking disk space...{RESET}")

        try:
            import psutil

            disk_usage = psutil.disk_usage('/home/vdidur')
            free_gb = disk_usage.free / (1024 ** 3)
            percent_used = disk_usage.percent

            if free_gb > 50:
                self.log_success(f"✓ {free_gb:.1f} GB free ({100 - percent_used:.1f}% available)")
            elif free_gb > 10:
                self.log_warning(f"⚠ Only {free_gb:.1f} GB free ({100 - percent_used:.1f}% available)")
            else:
                self.log_issue(f"✗ Low disk space: {free_gb:.1f} GB free")

        except Exception as e:
            self.log_warning(f"⚠ Could not check disk space: {e}")

    def check_network_connectivity(self):
        """Check network connectivity"""
        print(f"\n{BLUE}Checking network connectivity...{RESET}")

        # Check if we can resolve GPORTAL
        try:
            import socket
            socket.gethostbyname('gportal.jaxa.jp')
            self.log_success("✓ Can resolve gportal.jaxa.jp")
        except:
            self.log_issue("✗ Cannot resolve gportal.jaxa.jp")

    def create_test_job(self):
        """Create a test job file"""
        test_job = {
            "job_id": f"diagnostic_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "function": "single_strip",
            "parameters": {
                "date": "2025-05-26",
                "file_index": 0
            },
            "status": "pending",
            "submitted_time": datetime.now().isoformat()
        }

        test_job_path = self.server_root / 'jobs' / 'pending' / f"{test_job['job_id']}.json"

        try:
            with open(test_job_path, 'w') as f:
                json.dump(test_job, f, indent=2)
            return test_job['job_id']
        except:
            return None

    def log_success(self, message):
        """Log success message"""
        print(f"{GREEN}{message}{RESET}")
        self.successes.append(message)

    def log_warning(self, message):
        """Log warning message"""
        print(f"{YELLOW}{message}{RESET}")
        self.warnings.append(message)

    def log_issue(self, message):
        """Log issue message"""
        print(f"{RED}{message}{RESET}")
        self.issues.append(message)

    def print_summary(self):
        """Print diagnostic summary"""
        print(f"\n{BLUE}{'=' * 80}")
        print("DIAGNOSTIC SUMMARY")
        print(f"{'=' * 80}{RESET}\n")

        print(f"{GREEN}✓ Successes: {len(self.successes)}{RESET}")
        print(f"{YELLOW}⚠ Warnings: {len(self.warnings)}{RESET}")
        print(f"{RED}✗ Issues: {len(self.issues)}{RESET}")

        if self.issues:
            print(f"\n{RED}Critical issues that need fixing:{RESET}")
            for issue in self.issues[:5]:  # Show first 5 issues
                print(f"  • {issue}")
            if len(self.issues) > 5:
                print(f"  ... and {len(self.issues) - 5} more issues")

        if not self.issues:
            print(f"\n{GREEN}✓ Server appears to be properly configured!{RESET}")

            # Offer to create test job
            print("\nWould you like to create a test job? (y/n): ", end='')
            response = input().strip().lower()
            if response == 'y':
                job_id = self.create_test_job()
                if job_id:
                    print(f"\n{GREEN}✓ Created test job: {job_id}{RESET}")
                    print("Run the job processor to test:")
                    print(f"  python {self.server_root}/scripts/job_processor.py")
                else:
                    print(f"{RED}✗ Failed to create test job{RESET}")
        else:
            print(f"\n{YELLOW}Please fix the issues above before running the server.{RESET}")


if __name__ == "__main__":
    diagnostic = ServerDiagnostic()
    diagnostic.run_all_checks()