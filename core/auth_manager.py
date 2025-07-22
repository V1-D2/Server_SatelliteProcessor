"""
Authentication manager for gportal credentials
"""

import pathlib
import os


class AuthManager:
    """Manages gportal authentication credentials"""

    def __init__(self):
        self.config_dir = pathlib.Path(__file__).parent.parent / "config"
        self.credentials_file = self.config_dir / "credentials.txt"
        self.config_dir.mkdir(exist_ok=True)

    def has_credentials(self):
        """Check if credentials file exists and is not empty"""
        if self.credentials_file.exists():
            try:
                with open(self.credentials_file, 'r') as f:
                    content = f.read().strip()
                    return len(content) > 0
            except:
                return False
        return False

    def get_credentials(self):
        """Get stored credentials"""
        if not self.has_credentials():
            return None, None

        try:
            with open(self.credentials_file, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    username = lines[0].strip()
                    password = lines[1].strip()
                    return username, password
        except:
            pass

        return None, None

    def save_credentials(self, username, password):
        """Save credentials to file"""
        try:
            with open(self.credentials_file, 'w') as f:
                f.write(f"{username}\n{password}")
            return True
        except Exception as e:
            print(f"Error saving credentials: {e}")
            return False

    def test_credentials(self, username, password):
        """Test credentials with gportal API"""
        try:
            # Import gportal module
            import gportal

            # Set credentials
            gportal.username = username
            gportal.password = password

            # Try to access datasets to test authentication
            # This will raise an exception if credentials are invalid
            datasets = gportal.datasets()

            # If we got here, credentials are valid
            return True

        except Exception as e:
            # Authentication failed
            print(f"Authentication test failed: {e}")
            return False

    def clear_credentials(self):
        """Clear stored credentials"""
        try:
            if self.credentials_file.exists():
                self.credentials_file.unlink()
            return True
        except:
            return False