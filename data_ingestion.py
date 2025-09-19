#!/usr/bin/env python3
"""
Data Ingestion Script for Oceanographic NetCDF Files
Reads NOAA GADR Argo float data from NetCDF files and ingests into PostgreSQL database
"""

import asyncio
import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Tuple
import argparse
from dataclasses import dataclass
import uuid

# Third-party imports
import netCDF4 as nc
import numpy as np
import asyncpg
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    host: str
    port: int
    database: str
    username: str
    password: str
    
    @classmethod
    def from_env(cls):
        return cls(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "oceanographic_data"),
            username=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "")
        )
    
    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

class NetCDFIngester:
    """Main class for ingesting NetCDF data into PostgreSQL"""
    
    def __init__(self, db_config: DatabaseConfig):
        self.db_config = db_config
        self.connection = None
        self.stats = {
            'files_processed': 0,
            'platforms_created': 0,
            'profiles_created': 0,
            'measurements_created': 0,
            'errors': 0,
            'skipped': 0
        }
    
    async def connect(self):
        """Establish database connection"""
        try:
            self.connection = await asyncpg.connect(self.db_config.connection_string)
            logger.info("✅ Database connection established")
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            raise
    
    async def disconnect(self):
        """Close database connection"""
        if self.connection:
            await self.connection.close()
            logger.info("📕 Database connection closed")
    
    def extract_platform_data(self, dataset: nc.Dataset, filepath: str) -> Dict[str, Any]:
        """Extract platform information from NetCDF dataset"""
        platform_data = {}
        
        # Extract platform number from various possible sources
        platform_number = None
        if 'PLATFORM_NUMBER' in dataset.variables:
            platform_number = str(dataset.variables['PLATFORM_NUMBER'][:].tobytes().decode('utf-8').strip())
        elif hasattr(dataset, 'platform_number'):
            platform_number = str(dataset.platform_number).strip()
        elif hasattr(dataset, 'wmo_inst_type'):
            # Sometimes platform info is in global attributes
            platform_number = str(dataset.wmo_inst_type).strip()
        
        # If still no platform number, extract from filename
        if not platform_number or platform_number == '':
            filename = Path(filepath).stem
            # Extract platform ID from filename pattern like nodc_D53546_001
            parts = filename.split('_')
            if len(parts) >= 2:
                platform_number = parts[1]  # D53546
        
        platform_data['platform_number'] = platform_number
        
        # Extract other platform metadata
        platform_data['platform_type'] = getattr(dataset, 'platform_type', None)
        platform_data['float_serial'] = getattr(dataset, 'float_serial_no', None)
        platform_data['project_name'] = getattr(dataset, 'project_name', None)
        platform_data['pi_name'] = getattr(dataset, 'pi_name', None)
        platform_data['institution'] = getattr(dataset, 'institution', None)
        platform_data['data_center'] = getattr(dataset, 'data_centre', None)
        platform_data['wmo_inst_type'] = getattr(dataset, 'wmo_inst_type', None)
        platform_data['firmware_version'] = getattr(dataset, 'firmware_version', None)
        
        # Clean up string values
        for key, value in platform_data.items():
            if isinstance(value, (bytes, np.bytes_)):
                platform_data[key] = value.decode('utf-8').strip()
            elif isinstance(value, str):
                platform_data[key] = value.strip() if value.strip() else None
        
        return platform_data
    
    def extract_profile_data(self, dataset: nc.Dataset, platform_id: str) -> List[Dict[str, Any]]:
        """Extract profile information from NetCDF dataset"""
        profiles = []
        
        # Get number of profiles
        n_prof = dataset.dimensions.get('N_PROF', 1)
        
        for prof_idx in range(n_prof.size if hasattr(n_prof, 'size') else n_prof):
            profile_data = {
                'platform_id': platform_id,
                'cycle_number': self.safe_extract(dataset, 'cycle_number', prof_idx, int),
            }
            
            # Extract date/time information
            if 'juld' in dataset.variables:
                juld = self.safe_extract(dataset, 'juld', prof_idx, float)
                if juld and not np.isnan(juld):
                    # Convert JULD (Julian date) to datetime
                    # JULD reference date is typically 1950-01-01
                    reference_date = datetime(1950, 1, 1, tzinfo=timezone.utc)
                    profile_date = reference_date + timedelta(days=float(juld))
                    # Remove microseconds to avoid precision issues
                    profile_date = profile_date.replace(microsecond=0)
                    # Ensure timezone-aware datetime
                    if profile_date.tzinfo is None:
                        profile_date = profile_date.replace(tzinfo=timezone.utc)
                    profile_data['profile_date'] = profile_date
                    profile_data['juld'] = juld
            elif 'JULD' in dataset.variables:
                juld = self.safe_extract(dataset, 'JULD', prof_idx, float)
                if juld and not np.isnan(juld):
                    # Convert JULD (Julian date) to datetime
                    # JULD reference date is typically 1950-01-01
                    reference_date = datetime(1950, 1, 1, tzinfo=timezone.utc)
                    profile_date = reference_date + timedelta(days=float(juld))
                    # Remove microseconds to avoid precision issues
                    profile_date = profile_date.replace(microsecond=0)
                    # Ensure timezone-aware datetime
                    if profile_date.tzinfo is None:
                        profile_date = profile_date.replace(tzinfo=timezone.utc)
                    profile_data['profile_date'] = profile_date
                    profile_data['juld'] = juld
            
            # If no date from JULD, try to extract from filename or use current date
            if 'profile_date' not in profile_data:
                profile_data['profile_date'] = datetime.now(timezone.utc)
            
            # Geographic location
            profile_data['latitude'] = self.safe_extract(dataset, 'latitude', prof_idx, float)
            profile_data['longitude'] = self.safe_extract(dataset, 'longitude', prof_idx, float)
            
            # Other profile metadata
            profile_data['direction'] = self.safe_extract(dataset, 'direction', prof_idx, str)
            profile_data['data_mode'] = self.safe_extract(dataset, 'data_mode', prof_idx, str)
            profile_data['position_qc'] = self.safe_extract(dataset, 'position_qc', prof_idx, int)
            profile_data['juld_qc'] = self.safe_extract(dataset, 'juld_qc', prof_idx, int)
            profile_data['juld_location'] = self.safe_extract(dataset, 'juld_location', prof_idx, float)
            
            # Quality control flags
            profile_data['profile_pres_qc'] = self.safe_extract(dataset, 'profile_pres_qc', prof_idx, str)
            profile_data['profile_temp_qc'] = self.safe_extract(dataset, 'profile_temp_qc', prof_idx, str)
            profile_data['profile_psal_qc'] = self.safe_extract(dataset, 'profile_psal_qc', prof_idx, str)
            
            # Data processing metadata
            profile_data['data_state_indicator'] = self.safe_extract(dataset, 'data_state_indicator', prof_idx, str)
            profile_data['config_mission_number'] = self.safe_extract(dataset, 'config_mission_number', prof_idx, int)
            profile_data['positioning_system'] = self.safe_extract(dataset, 'positioning_system', prof_idx, str)
            profile_data['vertical_sampling_scheme'] = self.safe_extract(dataset, 'vertical_sampling_scheme', prof_idx, str)
            profile_data['dc_reference'] = self.safe_extract(dataset, 'dc_reference', prof_idx, str)
            
            # Generate geohash for spatial indexing
            if profile_data['latitude'] and profile_data['longitude']:
                profile_data['geohash'] = self.generate_geohash(
                    profile_data['latitude'], 
                    profile_data['longitude']
                )
            
            profiles.append(profile_data)
        
        return profiles
    
    def extract_measurements(self, dataset: nc.Dataset, profile_id: str, prof_idx: int = 0) -> List[Dict[str, Any]]:
        """Extract measurement data from NetCDF dataset"""
        measurements = []
        
        # Get number of levels
        n_levels = dataset.dimensions.get('n_levels', 0)
        if hasattr(n_levels, 'size'):
            n_levels = n_levels.size
        
        logger.debug(f"Number of levels: {n_levels}")
        
        for level_idx in range(n_levels):
            measurement = {
                'profile_id': profile_id,
            }
            
            # Pressure measurements (primary coordinate)
            measurement['pressure'] = self.safe_extract_2d(dataset, 'pres', prof_idx, level_idx, float)
            measurement['pressure_qc'] = self.safe_extract_2d(dataset, 'pres_qc', prof_idx, level_idx, int)
            measurement['pressure_adjusted'] = self.safe_extract_2d(dataset, 'pres_adjusted', prof_idx, level_idx, float)
            measurement['pressure_adjusted_qc'] = self.safe_extract_2d(dataset, 'pres_adjusted_qc', prof_idx, level_idx, int)
            measurement['pressure_adjusted_error'] = self.safe_extract_2d(dataset, 'pres_adjusted_error', prof_idx, level_idx, float)
            
            # Temperature measurements
            measurement['temperature'] = self.safe_extract_2d(dataset, 'temp', prof_idx, level_idx, float)
            measurement['temperature_qc'] = self.safe_extract_2d(dataset, 'temp_qc', prof_idx, level_idx, int)
            measurement['temperature_adjusted'] = self.safe_extract_2d(dataset, 'temp_adjusted', prof_idx, level_idx, float)
            measurement['temperature_adjusted_qc'] = self.safe_extract_2d(dataset, 'temp_adjusted_qc', prof_idx, level_idx, int)
            measurement['temperature_adjusted_error'] = self.safe_extract_2d(dataset, 'temp_adjusted_error', prof_idx, level_idx, float)
            
            # Salinity measurements
            measurement['salinity'] = self.safe_extract_2d(dataset, 'psal', prof_idx, level_idx, float)
            measurement['salinity_qc'] = self.safe_extract_2d(dataset, 'psal_qc', prof_idx, level_idx, int)
            measurement['salinity_adjusted'] = self.safe_extract_2d(dataset, 'psal_adjusted', prof_idx, level_idx, float)
            measurement['salinity_adjusted_qc'] = self.safe_extract_2d(dataset, 'psal_adjusted_qc', prof_idx, level_idx, int)
            measurement['salinity_adjusted_error'] = self.safe_extract_2d(dataset, 'psal_adjusted_error', prof_idx, level_idx, float)
            
            # Skip measurements with no valid data
            has_data = any([
                measurement['pressure'] is not None,
                measurement['temperature'] is not None,
                measurement['salinity'] is not None
            ])
            
            if has_data:
                measurements.append(measurement)
        
        return measurements
    
    def safe_extract(self, dataset: nc.Dataset, var_name: str, index: int, dtype) -> Any:
        """Safely extract a single value from NetCDF variable"""
        try:
            if var_name not in dataset.variables:
                return None
            
            var = dataset.variables[var_name]
            
            if len(var.shape) == 0:  # Scalar
                value = var[...]
            elif len(var.shape) == 1:  # 1D array
                if index < var.shape[0]:
                    value = var[index]
                else:
                    return None
            else:
                return None
            
            # Handle MaskedArray
            if isinstance(value, np.ma.MaskedArray):
                if value.mask.any() if hasattr(value.mask, 'any') else value.mask:
                    return None
                value = value.data.item() if hasattr(value.data, 'item') else value.data
            
            # Handle fill values and NaN
            if hasattr(var, '_FillValue') and value == var._FillValue:
                return None
            if np.isnan(value) if isinstance(value, (float, np.floating)) else False:
                return None
            
            # Convert to appropriate type
            if dtype == str and isinstance(value, (bytes, np.bytes_)):
                return value.decode('utf-8').strip()
            elif dtype == str:
                return str(value).strip() if str(value).strip() else None
            elif dtype in (int, float):
                return dtype(value)
            
            return value
            
        except (IndexError, ValueError, TypeError) as e:
            logger.debug(f"Error extracting {var_name}[{index}]: {e}")
            return None
    
    def safe_extract_2d(self, dataset: nc.Dataset, var_name: str, prof_idx: int, level_idx: int, dtype) -> Any:
        """Safely extract a value from 2D NetCDF variable"""
        try:
            if var_name not in dataset.variables:
                return None
            
            var = dataset.variables[var_name]
            
            if len(var.shape) >= 2:
                if prof_idx < var.shape[0] and level_idx < var.shape[1]:
                    value = var[prof_idx, level_idx]
                else:
                    return None
            else:
                return None
            
            # Handle MaskedArray
            if isinstance(value, np.ma.MaskedArray):
                if value.mask.any() if hasattr(value.mask, 'any') else value.mask:
                    return None
                value = value.data.item() if hasattr(value.data, 'item') else value.data
            
            # Handle fill values and NaN
            if hasattr(var, '_FillValue') and value == var._FillValue:
                return None
            if np.isnan(value) if isinstance(value, (float, np.floating)) else False:
                return None
            
            # Convert to appropriate type
            if dtype in (int, float):
                return dtype(value)
            
            return value
            
        except (IndexError, ValueError, TypeError) as e:
            logger.debug(f"Error extracting {var_name}[{prof_idx},{level_idx}]: {e}")
            return None
    
    def generate_geohash(self, lat: float, lon: float, precision: int = 7) -> str:
        """Generate a simple geohash for spatial indexing"""
        # This is a simplified geohash implementation
        # For production, consider using the python-geohash library
        try:
            lat_str = f"{lat:.6f}".replace('.', '').replace('-', 'n')[:precision]
            lon_str = f"{lon:.6f}".replace('.', '').replace('-', 'n')[:precision]
            return f"{lat_str}_{lon_str}"
        except:
            return None
    
    async def upsert_platform(self, platform_data: Dict[str, Any]) -> str:
        """Insert or update platform data and return platform ID"""
        try:
            # Check if platform exists
            query = "SELECT id FROM platforms WHERE platform_number = $1"
            platform_id = await self.connection.fetchval(query, platform_data['platform_number'])
            
            if platform_id:
                # Update existing platform
                update_query = """
                    UPDATE platforms SET
                        platform_type = COALESCE($2, platform_type),
                        float_serial = COALESCE($3, float_serial),
                        project_name = COALESCE($4, project_name),
                        pi_name = COALESCE($5, pi_name),
                        institution = COALESCE($6, institution),
                        data_center = COALESCE($7, data_center),
                        wmo_inst_type = COALESCE($8, wmo_inst_type),
                        firmware_version = COALESCE($9, firmware_version),
                        updated_at = NOW()
                    WHERE platform_number = $1
                    RETURNING id
                """
                platform_id = await self.connection.fetchval(
                    update_query,
                    platform_data['platform_number'],
                    platform_data.get('platform_type'),
                    platform_data.get('float_serial'),
                    platform_data.get('project_name'),
                    platform_data.get('pi_name'),
                    platform_data.get('institution'),
                    platform_data.get('data_center'),
                    platform_data.get('wmo_inst_type'),
                    platform_data.get('firmware_version')
                )
            else:
                # Insert new platform
                platform_id = str(uuid.uuid4())  # Generate UUID
                insert_query = """
                    INSERT INTO platforms (
                        id, platform_number, platform_type, float_serial, project_name,
                        pi_name, institution, data_center, wmo_inst_type, firmware_version,
                        created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW(), NOW())
                    RETURNING id
                """
                returned_id = await self.connection.fetchval(
                    insert_query,
                    platform_id,
                    platform_data['platform_number'],
                    platform_data.get('platform_type'),
                    platform_data.get('float_serial'),
                    platform_data.get('project_name'),
                    platform_data.get('pi_name'),
                    platform_data.get('institution'),
                    platform_data.get('data_center'),
                    platform_data.get('wmo_inst_type'),
                    platform_data.get('firmware_version')
                )
                self.stats['platforms_created'] += 1
            
            return platform_id
            
        except Exception as e:
            logger.error(f"Error upserting platform {platform_data.get('platform_number')}: {e}")
            raise
    
    async def insert_profile(self, profile_data: Dict[str, Any]) -> str:
        """Insert profile data and return profile ID"""
        try:
            # Debug print
            logger.debug(f"Inserting profile with date: {profile_data.get('profile_date')} (type: {type(profile_data.get('profile_date'))})")
            logger.debug(f"Lat/Lon: {profile_data.get('latitude')}/{profile_data.get('longitude')} (types: {type(profile_data.get('latitude'))}, {type(profile_data.get('longitude'))})")
            
            # Skip duplicate check for now to isolate the issue
            # existing_id = await self.connection.fetchval(check_query, ...)
            
            # Insert new profile
            profile_id = str(uuid.uuid4())  # Generate UUID
            insert_query = """
                INSERT INTO profiles (
                    id, platform_id, cycle_number, profile_date, direction, data_mode,
                    latitude, longitude, position_qc, profile_pres_qc, profile_temp_qc, profile_psal_qc,
                    juld, juld_qc, juld_location, data_state_indicator, config_mission_number,
                    positioning_system, vertical_sampling_scheme, dc_reference, geohash,
                    created_at, updated_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21,
                    NOW(), NOW()
                )
                RETURNING id
            """
            
            # Debug all the parameters being passed
            params = [
                profile_id,
                profile_data['platform_id'],
                profile_data.get('cycle_number'),
                profile_data.get('profile_date').replace(tzinfo=None) if profile_data.get('profile_date') else None,  # Remove timezone for PostgreSQL
                profile_data.get('direction'),
                profile_data.get('data_mode'),
                profile_data.get('latitude'),
                profile_data.get('longitude'),
                profile_data.get('position_qc'),
                profile_data.get('profile_pres_qc'),
                profile_data.get('profile_temp_qc'),
                profile_data.get('profile_psal_qc'),
                profile_data.get('juld'),
                profile_data.get('juld_qc'),
                profile_data.get('juld_location'),
                profile_data.get('data_state_indicator'),
                profile_data.get('config_mission_number'),
                profile_data.get('positioning_system'),
                profile_data.get('vertical_sampling_scheme'),
                profile_data.get('dc_reference'),
                profile_data.get('geohash')
            ]
            
            for i, param in enumerate(params, 1):
                logger.debug(f"Parameter ${i}: {param} (type: {type(param)})")
            
            returned_id = await self.connection.fetchval(insert_query, *params)
            
            self.stats['profiles_created'] += 1
            return profile_id
            
        except Exception as e:
            logger.error(f"Error inserting profile: {e}")
            raise
    
    async def insert_measurements_batch(self, measurements: List[Dict[str, Any]]):
        """Insert measurements in batch for better performance"""
        if not measurements:
            return
        
        try:
            # Prepare batch insert query
            insert_query = """
                INSERT INTO measurements (
                    id, profile_id, pressure, pressure_qc, pressure_adjusted, pressure_adjusted_qc, pressure_adjusted_error,
                    temp, temp_qc, temp_adjusted, temp_adjusted_qc, temp_adjusted_error,
                    psal, psal_qc, psal_adjusted, psal_adjusted_qc, psal_adjusted_error,
                    created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, NOW(), NOW())
            """
            
            # Prepare data for batch insert
            batch_data = []
            for m in measurements:
                batch_data.append((
                    str(uuid.uuid4()),  # Generate UUID for each measurement
                    m['profile_id'],
                    m.get('pressure'), m.get('pressure_qc'), m.get('pressure_adjusted'), 
                    m.get('pressure_adjusted_qc'), m.get('pressure_adjusted_error'),
                    m.get('temperature'), m.get('temperature_qc'), m.get('temperature_adjusted'),
                    m.get('temperature_adjusted_qc'), m.get('temperature_adjusted_error'),
                    m.get('salinity'), m.get('salinity_qc'), m.get('salinity_adjusted'),
                    m.get('salinity_adjusted_qc'), m.get('salinity_adjusted_error')
                ))
            
            # Execute batch insert
            await self.connection.executemany(insert_query, batch_data)
            self.stats['measurements_created'] += len(measurements)
            
        except Exception as e:
            logger.error(f"Error inserting measurements batch: {e}")
            raise
    
    async def process_file(self, filepath: str) -> bool:
        """Process a single NetCDF file"""
        logger.info(f"📁 Processing file: {os.path.basename(filepath)}")
        
        try:
            with nc.Dataset(filepath, 'r') as dataset:
                # Extract platform data
                platform_data = self.extract_platform_data(dataset, filepath)
                if not platform_data.get('platform_number'):
                    logger.warning(f"⚠️  No platform number found in {filepath}, skipping")
                    self.stats['skipped'] += 1
                    return False
                
                # Upsert platform
                platform_id = await self.upsert_platform(platform_data)
                logger.debug(f"Platform ID: {platform_id}")
                
                # Extract and insert profiles
                profiles = self.extract_profile_data(dataset, platform_id)
                
                for prof_idx, profile_data in enumerate(profiles):
                    profile_id = await self.insert_profile(profile_data)
                    logger.debug(f"Profile ID: {profile_id}")
                    
                    # Extract and insert measurements
                    measurements = self.extract_measurements(dataset, profile_id, prof_idx)
                    logger.debug(f"Extracted {len(measurements)} measurements for profile {prof_idx}")
                    if measurements:
                        await self.insert_measurements_batch(measurements)
                        logger.debug(f"Inserted {len(measurements)} measurements")
                
                self.stats['files_processed'] += 1
                logger.info(f"✅ Successfully processed {os.path.basename(filepath)}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error processing {filepath}: {e}")
            self.stats['errors'] += 1
            return False
    
    async def process_directory(self, directory: str, pattern: str = "*.nc"):
        """Process all NetCDF files in a directory and subdirectories"""
        logger.info(f"🔍 Scanning directory: {directory}")
        
        nc_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.nc'):
                    nc_files.append(os.path.join(root, file))
        
        logger.info(f"📊 Found {len(nc_files)} NetCDF files")
        
        # Process files
        for filepath in nc_files:
            await self.process_file(filepath)
            
            # Print progress every 10 files
            if self.stats['files_processed'] % 10 == 0:
                logger.info(f"📈 Progress: {self.stats['files_processed']}/{len(nc_files)} files processed")
    
    def print_stats(self):
        """Print ingestion statistics"""
        print("\n" + "="*50)
        print("📊 INGESTION STATISTICS")
        print("="*50)
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Platforms created: {self.stats['platforms_created']}")
        print(f"Profiles created: {self.stats['profiles_created']}")
        print(f"Measurements created: {self.stats['measurements_created']}")
        print(f"Errors: {self.stats['errors']}")
        print(f"Skipped: {self.stats['skipped']}")
        print("="*50)

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Ingest NOAA Argo NetCDF data into PostgreSQL database')
    parser.add_argument('input_path', help='Path to NetCDF file or directory containing NetCDF files')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for measurements (default: 100)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load database configuration
    try:
        db_config = DatabaseConfig.from_env()
        logger.info(f"🔗 Connecting to database: {db_config.host}:{db_config.port}/{db_config.database}")
    except Exception as e:
        logger.error(f"❌ Failed to load database configuration: {e}")
        return
    
    # Initialize ingester
    ingester = NetCDFIngester(db_config)
    
    try:
        # Connect to database
        await ingester.connect()
        
        # Process input
        input_path = Path(args.input_path)
        if input_path.is_file():
            await ingester.process_file(str(input_path))
        elif input_path.is_dir():
            await ingester.process_directory(str(input_path))
        else:
            logger.error(f"❌ Invalid input path: {input_path}")
            return
        
        # Print statistics
        ingester.print_stats()
        
    except KeyboardInterrupt:
        logger.info("⏹️  Ingestion interrupted by user")
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
    finally:
        await ingester.disconnect()

if __name__ == "__main__":
    asyncio.run(main())