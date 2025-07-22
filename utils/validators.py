"""
Validators for user input
"""

import datetime
from typing import Tuple, Optional


class DateValidator:
    """Validates date inputs"""

    def __init__(self):
        # Minimum date for AMSR-2 data
        self.min_date = datetime.date(2013, 1, 1)

    def validate_date(self, date_str: str) -> Tuple[bool, str, Optional[datetime.date]]:
        """
        Validate date string in MM/DD/YYYY format

        Args:
            date_str: Date string to validate

        Returns:
            Tuple of (is_valid, error_message, date_object)
        """
        # Check if empty
        if not date_str or not date_str.strip():
            return False, "Please enter a date", None

        # Try to parse date
        try:
            # Parse MM/DD/YYYY format
            date_obj = datetime.datetime.strptime(date_str.strip(), "%m/%d/%Y").date()

        except ValueError:
            try:
                # Try alternative formats
                date_obj = datetime.datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
            except ValueError:
                return False, "Invalid date format. Please use MM/DD/YYYY", None

        # Check if date is in valid range
        today = datetime.date.today()

        if date_obj < self.min_date:
            return False, f"Date must be after {self.min_date.strftime('%m/%d/%Y')}", None

        if date_obj > today:
            return False, "Date cannot be in the future", None

        # Valid date
        return True, "", date_obj

    def format_date_for_display(self, date_obj: datetime.date) -> str:
        """Format date for display in MM/DD/YYYY format"""
        return date_obj.strftime("%m/%d/%Y")

    def format_date_for_api(self, date_obj: datetime.date) -> str:
        """Format date for API in YYYY-MM-DD format"""
        return date_obj.strftime("%Y-%m-%d")

    def format_date_for_filename(self, date_obj: datetime.date) -> str:
        """Format date for filename in YYYYMMDD format"""
        return date_obj.strftime("%Y%m%d")

    def parse_any_format(self, date_str: str) -> Optional[datetime.date]:
        """
        Try to parse date in various formats

        Args:
            date_str: Date string in unknown format

        Returns:
            Date object or None
        """
        formats = [
            "%m/%d/%Y",  # MM/DD/YYYY
            "%Y-%m-%d",  # YYYY-MM-DD
            "%Y/%m/%d",  # YYYY/MM/DD
            "%d/%m/%Y",  # DD/MM/YYYY
            "%m-%d-%Y",  # MM-DD-YYYY
            "%d-%m-%Y",  # DD-MM-YYYY
            "%Y%m%d",  # YYYYMMDD
            "%m/%d/%y",  # MM/DD/YY
            "%d.%m.%Y",  # DD.MM.YYYY
        ]

        for fmt in formats:
            try:
                return datetime.datetime.strptime(date_str.strip(), fmt).date()
            except ValueError:
                continue

        return None

    def get_date_range_for_day(self, date_obj: datetime.date) -> Tuple[str, str]:
        """
        Get start and end datetime strings for a given date

        Args:
            date_obj: Date object

        Returns:
            Tuple of (start_datetime, end_datetime) in ISO format
        """
        start_dt = datetime.datetime.combine(date_obj, datetime.time.min)
        end_dt = datetime.datetime.combine(date_obj, datetime.time.max)

        return start_dt.isoformat(), end_dt.isoformat()


class FileValidator:
    """Validates file inputs"""

    @staticmethod
    def validate_file_selection(file_index: str, max_files: int) -> Tuple[bool, str, Optional[int]]:
        """
        Validate file selection index

        Args:
            file_index: User input for file index
            max_files: Maximum number of files available

        Returns:
            Tuple of (is_valid, error_message, index)
        """
        # Check if empty
        if not file_index or not file_index.strip():
            return False, "Please select a file", None

        # Try to parse as integer
        try:
            index = int(file_index.strip())
        except ValueError:
            return False, "Please enter a valid number", None

        # Check range
        if index < 1:
            return False, "File number must be greater than 0", None

        if index > max_files:
            return False, f"File number must be between 1 and {max_files}", None

        # Valid selection (convert to 0-based index)
        return True, "", index - 1

    @staticmethod
    def validate_orbit_type(orbit_type: str) -> bool:
        """Validate orbit type selection"""
        return orbit_type in ["A", "D"]

    @staticmethod
    def validate_pole(pole: str) -> bool:
        """Validate pole selection"""
        return pole in ["N", "S"]