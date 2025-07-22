# Code Integration Instructions for SatelliteProcessor

## Overview
This document explains how to integrate your existing code into the SatelliteProcessor application.

## Integration Points

### 1. GPORTAL Authentication (core/gportal_client.py)

The application already includes basic GPORTAL setup. If you need to modify the authentication or dataset selection:

```python
# In core/gportal_client.py, modify the _setup_dataset method:
def _setup_dataset(self):
    """Set up dataset ID"""
    try:
        # Your specific dataset configuration here
        _DS = gportal.datasets()["GCOM-W/AMSR2"]["LEVEL1"]
        self.DS_L1B_TB = _DS["L1B-Brightness temperature（TB）"][0]
    except Exception as e:
        print(f"Error setting up dataset: {e}")
        raise
```

### 2. Polar Image Creation (core/image_processor.py)

The polar image creation logic from your code (paste-3.txt) has been integrated. The main processing flow is:

1. Create EASE-Grid 2.0 grids
2. Process each HDF5 file
3. Extract temperature data with scale factor
4. Transform coordinates
5. Accumulate data in grid
6. Fill holes intelligently
7. Save as color and grayscale images

**No changes needed** - your algorithm is fully implemented.

### 3. Temperature Data Extraction (core/data_handler.py)

The temperature extraction follows your pattern from paste-4.txt:

```python
# Scale factor is automatically applied:
scale_factor = h5[var_name].attrs.get("SCALE FACTOR", 1.0)
temp_data = raw_data * scale_factor
```

### 4. Grayscale Image Generation

To implement your specific grayscale conversion method, modify the `tensor2img` method in `core/image_processor.py`:

```python
def tensor2img(self, tensor_list):
    """
    Convert tensor to grayscale image
    Replace this with your specific implementation
    """
    if len(tensor_list) > 0:
        data = tensor_list[0]
        # Your specific grayscale conversion logic here
        return self._normalize_to_uint8(data)
    return None
```

### 5. File Organization

The application follows your requested organization:
- Downloads go to `temp/` directory
- Results are saved to user-selected output directory
- Subdirectories are created as: `{date}-{A/D}-{N/S}/`
- Temp files are automatically cleaned after processing

## Adding Your Enhancement Models (Future)

When your 8x enhancement models are ready:

### For Function 3 (Single File Enhancement):

1. In `gui/function_windows.py`, find the `Enhance8xWindow` class
2. Replace the placeholder `process_single_strip` method
3. Add your model loading and inference code

### For Function 4 (Enhanced Polar Circle):

1. In `gui/function_windows.py`, find the `PolarEnhanced8xWindow` class
2. Implement the full processing pipeline:
   - Download all files for the date
   - Enhance each file individually
   - Adjust coordinate arrays (multiply by 8)
   - Create combined polar image

## Important Notes

1. **Scale Factor**: Always applied automatically when extracting temperature
2. **Missing Data**: Represented as NaN in arrays
3. **Coordinate Systems**: Using official EASE-Grid 2.0 North (EPSG:6931)
4. **File Formats**: 
   - Input: HDF5 files from GPORTAL
   - Output: PNG images + NPZ temperature arrays

## Testing Your Integration

1. Start with a known good date (e.g., 2025-05-26)
2. Test Function 2 (Single Strip) first - it's simpler
3. Then test Function 1 (Polar Circle)
4. Verify outputs match your expectations

## Debugging Tips

- Check `temp/` directory during processing to see downloaded files
- Look for error messages in the console
- Verify scale factors are being applied correctly
- Ensure coordinate transformations are working

## Need Help?

If the integration doesn't work as expected:
1. Check that all dependencies are installed
2. Verify GPORTAL credentials are correct
3. Ensure date has available data
4. Check console output for specific error messages