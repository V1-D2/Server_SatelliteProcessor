"""
Progress display utilities for CLI
"""

import sys
import time
import threading
from contextlib import contextmanager
from typing import Optional

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class ProgressDisplay:
    """Handles various progress indicators for CLI"""

    def __init__(self):
        self.spinner_active = False
        self.spinner_thread = None

    def show_banner(self):
        """Display application banner"""
        banner = """
        ╔════════════════════════════════════════════════════╗
        ║          SATELLITEPROCESSOR CLI v1.0.0             ║
        ║         AMSR-2 Satellite Data Processing           ║
        ╚════════════════════════════════════════════════════╝
        """
        print(banner)

    def info(self, message: str):
        """Display info message"""
        print(f"[INFO] {message}")

    def success(self, message: str):
        """Display success message"""
        print(f"[✓] {message}")

    def error(self, message: str):
        """Display error message"""
        print(f"[✗] {message}")

    def warning(self, message: str):
        """Display warning message"""
        print(f"[!] {message}")

    def start_spinner(self, message: str):
        """Start a spinning progress indicator"""
        self.spinner_active = True
        self.spinner_thread = threading.Thread(
            target=self._spin,
            args=(message,)
        )
        self.spinner_thread.daemon = True
        self.spinner_thread.start()

    def stop_spinner(self):
        """Stop the spinning progress indicator"""
        if self.spinner_active:
            self.spinner_active = False
            if self.spinner_thread:
                self.spinner_thread.join()
            # Clear the spinner line
            sys.stdout.write('\r' + ' ' * 80 + '\r')
            sys.stdout.flush()

    def _spin(self, message: str):
        """Spinner animation thread"""
        spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        # Fallback for terminals that don't support Unicode
        if not self._supports_unicode():
            spinner_chars = ['|', '/', '-', '\\']

        i = 0
        while self.spinner_active:
            char = spinner_chars[i % len(spinner_chars)]
            sys.stdout.write(f'\r{char} {message}')
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1

    def _supports_unicode(self) -> bool:
        """Check if terminal supports Unicode"""
        try:
            # Try to encode a Unicode character
            '✓'.encode(sys.stdout.encoding or 'ascii')
            return True
        except (UnicodeEncodeError, AttributeError):
            return False

    @contextmanager
    def create_progress_bar(self, total: int, description: str = "Processing"):
        """Create a progress bar context manager"""
        if TQDM_AVAILABLE:
            pbar = tqdm(
                total=total,
                desc=description,
                unit='files',
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
            try:
                yield pbar
            finally:
                pbar.close()
        else:
            # Fallback progress bar
            pbar = SimpleProgressBar(total, description)
            try:
                yield pbar
            finally:
                pbar.close()

    def show_step_progress(self, current: int, total: int, description: str):
        """Show step-based progress"""
        percentage = (current / total) * 100
        bar_length = 40
        filled_length = int(bar_length * current // total)

        bar = '█' * filled_length + '░' * (bar_length - filled_length)

        sys.stdout.write(f'\r{description}: [{bar}] {percentage:.1f}% ({current}/{total})')
        sys.stdout.flush()

        if current == total:
            print()  # New line when complete

    def show_download_progress(self, downloaded: int, total: int, filename: str):
        """Show download progress with size information"""
        if total > 0:
            percentage = (downloaded / total) * 100
            downloaded_mb = downloaded / (1024 * 1024)
            total_mb = total / (1024 * 1024)

            sys.stdout.write(
                f'\rDownloading {filename}: {downloaded_mb:.1f}/{total_mb:.1f} MB ({percentage:.1f}%)'
            )
        else:
            downloaded_mb = downloaded / (1024 * 1024)
            sys.stdout.write(f'\rDownloading {filename}: {downloaded_mb:.1f} MB')

        sys.stdout.flush()

    def show_processing_stages(self, stages: list):
        """Show multi-stage processing progress"""
        print("\nProcessing Stages:")
        print("-" * 50)

        for i, stage in enumerate(stages, 1):
            print(f"{i}. {stage}")

        print("-" * 50)

    def update_stage(self, stage_num: int, total_stages: int, stage_name: str, status: str = "Processing"):
        """Update current stage status"""
        print(f"\n[Stage {stage_num}/{total_stages}] {stage_name}: {status}")

    def show_statistics(self, stats: dict, title: str = "Statistics"):
        """Display statistics in a formatted way"""
        print(f"\n{title}:")
        print("=" * 50)

        for key, value in stats.items():
            # Format the key to be more readable
            formatted_key = key.replace('_', ' ').title()

            # Format the value based on type
            if isinstance(value, float):
                if value < 0.01 or value > 1000:
                    formatted_value = f"{value:.2e}"
                else:
                    formatted_value = f"{value:.2f}"
            elif isinstance(value, int):
                formatted_value = f"{value:,}"
            else:
                formatted_value = str(value)

            print(f"{formatted_key:.<30} {formatted_value}")

        print("=" * 50)


class SimpleProgressBar:
    """Simple progress bar for when tqdm is not available"""

    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.description = description
        self.current = 0

    def update(self, n: int = 1):
        """Update progress"""
        self.current += n
        self._display()

    def set_description(self, desc: str):
        """Update description"""
        self.description = desc
        self._display()

    def _display(self):
        """Display the progress bar"""
        if self.total > 0:
            percentage = (self.current / self.total) * 100
            bar_length = 30
            filled = int(bar_length * self.current // self.total)

            bar = '=' * filled + '>' + '-' * (bar_length - filled - 1)

            sys.stdout.write(
                f'\r{self.description}: [{bar}] {percentage:.0f}% ({self.current}/{self.total})'
            )
        else:
            sys.stdout.write(f'\r{self.description}: {self.current} items')

        sys.stdout.flush()

    def close(self):
        """Finish the progress bar"""
        print()  # New line