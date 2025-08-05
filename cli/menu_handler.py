"""
Menu handler for CLI interface
"""

import os
from typing import Optional


class MenuHandler:
    """Handles menu display and navigation"""

    def __init__(self):
        self.menu_width = 60

    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def show_main_menu(self) -> str:
        """Display main menu and get user choice"""
        self.clear_screen()

        print("=" * self.menu_width)
        print("SATELLITEPROCESSOR CLI - MAIN MENU".center(self.menu_width))
        print("=" * self.menu_width)
        print("\nAvailable Functions:")
        print("\n1. Polar Circle")
        print("   Create circular polar image from multiple satellite passes")
        print("\n2. Single Strip")
        print("   Process individual satellite data file")
        print("\n3. 8x Enhancement")
        print("   Enhance single strip quality by 8x using ML")
        print("\n4. 8x Enhanced Polar Circle")
        print("   Create 8x enhanced polar image using ML")
        print("\n5. Exit")
        print("   Close application")
        print("\n" + "=" * self.menu_width)

        choice = input("\nSelect function (1-5) or type 'exit': ").strip()
        return choice

    def get_orbit_type(self) -> Optional[str]:
        """Get orbit type selection"""
        print("\n=== Select Orbit Type ===")
        print("1. Ascending (A)")
        print("2. Descending (D)")

        while True:
            choice = input("\nSelect orbit type (1-2): ").strip()

            if choice == '1':
                return 'A'
            elif choice == '2':
                return 'D'
            elif choice.lower() in ['cancel', 'back']:
                return None
            else:
                print("Invalid choice. Please select 1 or 2.")

    def get_pole(self) -> Optional[str]:
        """Get pole selection"""
        print("\n=== Select Pole ===")
        print("1. North")
        print("2. South")

        while True:
            choice = input("\nSelect pole (1-2): ").strip()

            if choice == '1':
                return 'N'
            elif choice == '2':
                return 'S'
            elif choice.lower() in ['cancel', 'back']:
                return None
            else:
                print("Invalid choice. Please select 1 or 2.")

    def show_file_list(self, files: list, page_size: int = 20) -> Optional[int]:
        """Display paginated file list and get selection"""
        total_files = len(files)
        total_pages = (total_files + page_size - 1) // page_size
        current_page = 0

        while True:
            self.clear_screen()

            # Calculate page bounds
            start_idx = current_page * page_size
            end_idx = min(start_idx + page_size, total_files)

            # Display header
            print(f"\n=== Available Files (Page {current_page + 1}/{total_pages}) ===")
            print("-" * self.menu_width)

            # Display files for current page
            for i in range(start_idx, end_idx):
                print(f"{i + 1:3d}. {files[i]['name']}")

            print("-" * self.menu_width)

            # Display navigation options
            nav_options = []
            if current_page > 0:
                nav_options.append("'p' - Previous page")
            if current_page < total_pages - 1:
                nav_options.append("'n' - Next page")
            nav_options.append("'c' - Cancel")

            print(" | ".join(nav_options))
            print(f"\nTotal files: {total_files}")

            # Get user input
            choice = input(f"\nSelect file (1-{total_files}) or navigation option: ").strip().lower()

            if choice == 'p' and current_page > 0:
                current_page -= 1
            elif choice == 'n' and current_page < total_pages - 1:
                current_page += 1
            elif choice == 'c':
                return None
            else:
                try:
                    file_num = int(choice)
                    if 1 <= file_num <= total_files:
                        return file_num - 1  # Convert to 0-based index
                    else:
                        print(f"Please enter a number between 1 and {total_files}")
                        input("Press Enter to continue...")
                except ValueError:
                    print("Invalid input. Please enter a number or navigation option.")
                    input("Press Enter to continue...")

    def confirm_action(self, message: str) -> bool:
        """Get confirmation for an action"""
        while True:
            response = input(f"\n{message} (y/n): ").strip().lower()
            if response == 'y':
                return True
            elif response == 'n':
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no.")

    def show_summary(self, title: str, items: dict):
        """Display a summary of information"""
        print(f"\n=== {title} ===")
        print("-" * self.menu_width)

        for key, value in items.items():
            print(f"{key:.<25} {value}")

        print("-" * self.menu_width)