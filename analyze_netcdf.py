#!/usr/bin/env python3
"""
Analyze NOAA Argo NetCDF files to understand their structure and variables
"""
import netCDF4 as nc
import os
from pathlib import Path

def analyze_netcdf_file(filepath):
    """Analyze a single NetCDF file and extract variable information"""
    print(f"\n{'='*60}")
    print(f"Analyzing file: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    try:
        with nc.Dataset(filepath, 'r') as dataset:
            # Print dimensions
            print("\nDIMENSIONS:")
            for dim_name, dim in dataset.dimensions.items():
                print(f"  {dim_name}: {len(dim)}")
            
            # Print global attributes
            print("\nGLOBAL ATTRIBUTES (first 10):")
            attrs = list(dataset.ncattrs())[:10]
            for attr in attrs:
                value = getattr(dataset, attr)
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                print(f"  {attr}: {value}")
            
            # Print variables with their dimensions and some key attributes
            print("\nVARIABLES:")
            for var_name, var in dataset.variables.items():
                var_info = f"  {var_name}: {var.dimensions}"
                
                # Add shape information
                var_info += f" (shape: {var.shape})"
                
                # Add data type
                var_info += f" [{var.dtype}]"
                
                # Add long_name if available
                if hasattr(var, 'long_name'):
                    var_info += f" - {var.long_name}"
                elif hasattr(var, 'standard_name'):
                    var_info += f" - {var.standard_name}"
                
                # Add units if available
                if hasattr(var, 'units'):
                    var_info += f" [{var.units}]"
                
                print(var_info)
                
                # For measurement variables, show some sample data
                if var_name.upper() in ['TEMP', 'PSAL', 'PRES', 'DOXY', 'CNDC'] and len(var.shape) > 0:
                    try:
                        sample_data = var[:]
                        valid_data = sample_data[~sample_data.mask] if hasattr(sample_data, 'mask') else sample_data
                        if len(valid_data) > 0:
                            print(f"    Sample values: {valid_data[:5]}")
                    except:
                        print(f"    (Could not read sample data)")
            
            return {
                'filename': os.path.basename(filepath),
                'dimensions': dict(dataset.dimensions),
                'variables': list(dataset.variables.keys()),
                'global_attrs': dataset.ncattrs()
            }
    
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        return None

def main():
    """Analyze all NetCDF files in the sample_data directory"""
    sample_dir = Path("sample_data")
    
    if not sample_dir.exists():
        print("sample_data directory not found!")
        return
    
    nc_files = list(sample_dir.glob("*.nc"))
    if not nc_files:
        print("No NetCDF files found in sample_data directory!")
        return
    
    print(f"Found {len(nc_files)} NetCDF files to analyze")
    
    all_variables = set()
    file_analyses = []
    
    for nc_file in nc_files:
        result = analyze_netcdf_file(nc_file)
        if result:
            file_analyses.append(result)
            all_variables.update(result['variables'])
    
    # Summary of all variables found across files
    print(f"\n{'='*60}")
    print("SUMMARY: ALL VARIABLES FOUND ACROSS FILES")
    print(f"{'='*60}")
    print(f"Total unique variables: {len(all_variables)}")
    print("\nAll variables:")
    for var in sorted(all_variables):
        print(f"  {var}")
    
    # Compare with common oceanographic variables
    common_ocean_vars_upper = [
        'TEMP', 'TEMP_QC', 'TEMP_ADJUSTED', 'TEMP_ADJUSTED_QC', 'TEMP_ADJUSTED_ERROR',
        'PSAL', 'PSAL_QC', 'PSAL_ADJUSTED', 'PSAL_ADJUSTED_QC', 'PSAL_ADJUSTED_ERROR',
        'PRES', 'PRES_QC', 'PRES_ADJUSTED', 'PRES_ADJUSTED_QC', 'PRES_ADJUSTED_ERROR',
        'DOXY', 'DOXY_QC', 'DOXY_ADJUSTED', 'DOXY_ADJUSTED_QC', 'DOXY_ADJUSTED_ERROR',
        'CNDC', 'CNDC_QC', 'CNDC_ADJUSTED', 'CNDC_ADJUSTED_QC', 'CNDC_ADJUSTED_ERROR',
        'PH_IN_SITU_TOTAL', 'PH_IN_SITU_TOTAL_QC', 'PH_IN_SITU_TOTAL_ADJUSTED',
        'NITRATE', 'NITRATE_QC', 'NITRATE_ADJUSTED',
        'CHLA', 'CHLA_QC', 'CHLA_ADJUSTED',
        'BBP700', 'BBP700_QC', 'BBP700_ADJUSTED',
        'CDOM', 'TURBIDITY'
    ]
    
    # Check both uppercase and lowercase versions
    common_ocean_vars = common_ocean_vars_upper + [var.lower() for var in common_ocean_vars_upper]
    
    print(f"\n{'='*60}")
    print("COMPARISON WITH EXPECTED OCEANOGRAPHIC VARIABLES")
    print(f"{'='*60}")
    
    found_vars = []
    missing_vars = []
    
    for var in common_ocean_vars:
        if var in all_variables:
            found_vars.append(var)
        else:
            missing_vars.append(var)
    
    print(f"\nFOUND in dataset ({len(found_vars)}):")
    for var in found_vars:
        print(f"  ✓ {var}")
    
    print(f"\nNOT FOUND in dataset ({len(missing_vars)}):")
    for var in missing_vars:
        print(f"  ✗ {var}")

if __name__ == "__main__":
    main()