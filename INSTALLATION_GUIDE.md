# SatelliteProcessor - Installation and Usage Guide

## Overview
SatelliteProcessor is a desktop application for processing AMSR-2 satellite data from GPORTAL.

## Prerequisites

1. **Python 3.8 or higher** installed on your system
2. **GPORTAL account** with valid credentials
3. **Windows OS** (for .exe build, though the Python code is cross-platform)
4. **At least 10GB free disk space** for temporary files and outputs

## Installation Steps

### 1. Clone or Download the Project
```bash
# If using git
git clone <repository-url>
cd SatelliteProcessor

# Or download and extract the ZIP file
```

### 2. Install Python Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Install gportal module (if available via pip)
pip install gportal
```

**Note**: If `gportal` module is not available via pip, you'll need to install it manually according to the GPORTAL documentation.

### 3. Project Structure
Ensure your project has the following structure:
```
SatelliteProcessor/
├── main.py
├── gui/
├── core/
├── utils/
├── assets/
├── config/
├── temp/
├── requirements.txt
└── build_exe.bat
```

## Running the Application

### Option 1: Run from Python
```bash
python main.py
```

### Option 2: Build and Run Executable
```bash
# On Windows, run the build script
build_exe.bat

# The executable will be created in dist/SatelliteProcessor.exe
```

## First Time Setup

1. **Login Screen**: Enter your GPORTAL username and password
2. **Output Directory**: Select where to save processed data
3. **Main Menu**: Choose from available functions

## Using the Application

### Function 1: Polar Circle
1. Enter date in MM/DD/YYYY format
2. Select orbit type (Ascending/Descending)
3. Select pole (currently only North is implemented)
4. Click "Process"
5. Results saved to: `{output_dir}/{date}-{A/D}-{N/S}/`

### Function 2: Single Strip
1. Enter date in MM/DD/YYYY format
2. Click "Check Files" to see available data
3. Select a file from the list
4. Click "Process"
5. Results saved to: `{output_dir}/SingleStrip-{date}/`

### Function 3 & 4: Coming Soon
These features are placeholders for future enhancement functionality.

## Output Files

Each processing function creates:
- **Color image** (turbo colormap) - `.png`
- **Grayscale image** - `.png`
- **Temperature data** - `.npz` with NaN for missing values

## Troubleshooting

### Authentication Issues
- Verify your GPORTAL credentials are correct
- Check your internet connection
- Ensure GPORTAL service is accessible

### Missing Data
- Data is available from 01/01/2013 onwards
- Some dates may have no data due to satellite coverage

### File Not Found Errors
- Ensure all project directories exist
- Check file permissions
- Verify the gportal module is properly installed

### Memory Issues
- The application processes large satellite files
- Ensure at least 4GB RAM available
- Close other applications if needed

## Integration Points for User Code

The following files are designed for easy integration of user-provided code:

1. **core/gportal_client.py** - Line 15-25: Dataset setup
   - Replace with your specific dataset configuration

2. **core/image_processor.py** - Line 250-260: Grayscale conversion
   - Implement your specific `tensor2img` method here

3. **core/data_handler.py** - Temperature extraction logic
   - Modify if your data format differs

## Building for Distribution

To create a standalone executable:

1. Ensure all dependencies are installed
2. Place an icon file at `assets/icon.ico` (optional)
3. Run `build_exe.bat`
4. Distribute the entire `dist` folder contents

## Important Notes

- The application stores credentials in plain text (as requested)
- Temporary files are automatically cleaned after processing
- All dates/times are in UTC
- North pole processing is fully implemented; South pole is a placeholder

## Support

For issues related to:
- GPORTAL API: Consult GPORTAL documentation
- Application bugs: Check error messages and logs
- Enhancement features: Wait for future updates