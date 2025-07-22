"""
Create a simple icon for SatelliteProcessor application
"""

from PIL import Image, ImageDraw, ImageFont
import os


def create_icon():
    """Create a simple satellite-themed icon"""

    # Create a new image with blue background
    size = 256
    img = Image.new('RGBA', (size, size), color=(30, 50, 80, 255))
    draw = ImageDraw.Draw(img)

    # Draw a circle (representing Earth)
    earth_color = (100, 150, 200, 255)
    earth_radius = 80
    earth_center = (size // 2, size // 2)
    draw.ellipse(
        [earth_center[0] - earth_radius, earth_center[1] - earth_radius,
         earth_center[0] + earth_radius, earth_center[1] + earth_radius],
        fill=earth_color
    )

    # Draw satellite orbit (ellipse)
    orbit_color = (255, 255, 255, 200)
    draw.ellipse(
        [40, 100, size - 40, size - 100],
        outline=orbit_color,
        width=3
    )

    # Draw satellite (small rectangle)
    sat_color = (255, 255, 255, 255)
    sat_size = 20
    sat_x = size - 60
    sat_y = 60
    draw.rectangle(
        [sat_x - sat_size // 2, sat_y - sat_size // 2,
         sat_x + sat_size // 2, sat_y + sat_size // 2],
        fill=sat_color
    )

    # Draw solar panels
    panel_width = 30
    panel_height = 10
    # Left panel
    draw.rectangle(
        [sat_x - sat_size // 2 - panel_width, sat_y - panel_height // 2,
         sat_x - sat_size // 2, sat_y + panel_height // 2],
        fill=sat_color
    )
    # Right panel
    draw.rectangle(
        [sat_x + sat_size // 2, sat_y - panel_height // 2,
         sat_x + sat_size // 2 + panel_width, sat_y + panel_height // 2],
        fill=sat_color
    )

    # Add text
    try:
        # Try to use a built-in font
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        # Use default font if arial not available
        font = ImageFont.load_default()

    text = "SatProc"
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = (size - text_width) // 2
    text_y = size - 40

    draw.text((text_x, text_y), text, fill=(255, 255, 255, 255), font=font)

    # Save as ICO file
    os.makedirs("assets", exist_ok=True)

    # Create multiple sizes for ICO
    icon_sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]

    # Save as ICO with multiple resolutions
    img.save("assets/icon.ico", format="ICO", sizes=icon_sizes)

    # Also save as PNG for other uses
    img.save("assets/icon.png", format="PNG")

    print("Icon created successfully!")
    print("- assets/icon.ico (for Windows executable)")
    print("- assets/icon.png (for other uses)")


if __name__ == "__main__":
    create_icon()