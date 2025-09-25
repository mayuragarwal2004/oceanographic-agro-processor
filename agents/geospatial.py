"""
Geospatial Agent
Resolves location names to geographic coordinates and bounding boxes
Handles spatial queries and geographic filtering for oceanographic data
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from .base_agent import BaseAgent, AgentResult, LLMMessage, extract_json_string

# Add query-specific logging
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.query_logging import get_query_logger

class LocationType(Enum):
    """Types of geographic locations"""
    OCEAN = "ocean"
    SEA = "sea"
    GULF = "gulf"
    BAY = "bay"
    STRAIT = "strait"
    COORDINATES = "coordinates"
    BOUNDING_BOX = "bounding_box"
    UNKNOWN = "unknown"

@dataclass
class GeographicBounds:
    """Geographic bounding box coordinates"""
    north: float  # Latitude
    south: float  # Latitude
    east: float   # Longitude
    west: float   # Longitude
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'north': self.north,
            'south': self.south,
            'east': self.east,
            'west': self.west
        }

@dataclass
class Location:
    """Resolved location with coordinates"""
    name: str
    location_type: LocationType
    bounds: GeographicBounds
    center_lat: float
    center_lon: float
    confidence: float
    aliases: List[str] = None

@dataclass
class SpatialQuery:
    """Processed spatial query information"""
    locations: List[Location]
    spatial_operation: str  # intersects, within, contains, etc.
    resolution: str  # fine, medium, coarse
    projection: Optional[str] = None

class GeospatialAgent(BaseAgent):
    """Agent responsible for geographic location resolution and spatial queries"""
    
    def __init__(self, config):
        super().__init__(config, "geospatial")
        
        # Query-specific logging manager
        self.query_logger_manager = get_query_logger()
        self.current_query_logger = None
        
        # Predefined ocean/sea boundaries (approximate)
        self.predefined_locations = {
            'pacific ocean': Location(
                name='Pacific Ocean',
                location_type=LocationType.OCEAN,
                bounds=GeographicBounds(north=60, south=-60, east=-70, west=120),
                center_lat=0, center_lon=155,
                confidence=0.95,
                aliases=['pacific', 'north pacific', 'south pacific']
            ),
            'atlantic ocean': Location(
                name='Atlantic Ocean',
                location_type=LocationType.OCEAN,
                bounds=GeographicBounds(north=70, south=-60, east=20, west=-80),
                center_lat=5, center_lon=-30,
                confidence=0.95,
                aliases=['atlantic', 'north atlantic', 'south atlantic']
            ),
            'indian ocean': Location(
                name='Indian Ocean',
                location_type=LocationType.OCEAN,
                bounds=GeographicBounds(north=30, south=-60, east=120, west=20),
                center_lat=-15, center_lon=70,
                confidence=0.95,
                aliases=['indian']
            ),
            'arctic ocean': Location(
                name='Arctic Ocean',
                location_type=LocationType.OCEAN,
                bounds=GeographicBounds(north=90, south=60, east=180, west=-180),
                center_lat=75, center_lon=0,
                confidence=0.95,
                aliases=['arctic']
            ),
            'southern ocean': Location(
                name='Southern Ocean',
                location_type=LocationType.OCEAN,
                bounds=GeographicBounds(north=-50, south=-90, east=180, west=-180),
                center_lat=-70, center_lon=0,
                confidence=0.95,
                aliases=['antarctic ocean']
            ),
            'mediterranean sea': Location(
                name='Mediterranean Sea',
                location_type=LocationType.SEA,
                bounds=GeographicBounds(north=47, south=30, east=42, west=-6),
                center_lat=38.5, center_lon=18,
                confidence=0.95,
                aliases=['mediterranean']
            ),
            'caribbean sea': Location(
                name='Caribbean Sea',
                location_type=LocationType.SEA,
                bounds=GeographicBounds(north=22, south=9, east=-59, west=-87),
                center_lat=15.5, center_lon=-73,
                confidence=0.95,
                aliases=['caribbean']
            ),
            'gulf of mexico': Location(
                name='Gulf of Mexico',
                location_type=LocationType.GULF,
                bounds=GeographicBounds(north=31, south=18, east=-80, west=-98),
                center_lat=24.5, center_lon=-89,
                confidence=0.95,
                aliases=['gulf of mexico', 'gom']
            ),
            'bay of bengal': Location(
                name='Bay of Bengal',
                location_type=LocationType.BAY,
                bounds=GeographicBounds(north=23, south=5, east=100, west=80),
                center_lat=14, center_lon=90,
                confidence=0.95,
                aliases=['bay of bengal', 'bengal bay']
            )
        }
    
    async def process(self, input_data: Any, context: Dict[str, Any] = None) -> AgentResult:
        """Process location names and resolve to geographic coordinates"""
        
        # Use shared query logger from context, or create new one as fallback
        if context and 'current_query_logger' in context:
            self.current_query_logger = context['current_query_logger']
        else:
            # Fallback: create new logger (for backward compatibility)
            original_query = context.get('original_query', 'unknown_query') if context else 'unknown_query'
            session_id = context.get('session_id', None) if context else None
            self.current_query_logger = self.query_logger_manager.get_query_logger(original_query, session_id)
        
        try:
            # Log agent start
            self.current_query_logger.log_agent_start(self.agent_name, input_data)
            
            # Input can be a list of location names or a spatial query dict
            if isinstance(input_data, list):
                location_names = input_data
            elif isinstance(input_data, dict) and 'locations' in input_data:
                location_names = input_data['locations']
            else:
                error_msg = "Input must be a list of location names or dict with 'locations' key"
                self.current_query_logger.log_error(
                    self.agent_name,
                    ValueError(error_msg), "Input validation"
                )
                return AgentResult.error_result(self.agent_name, [error_msg])
            
            self.logger.info(f"Resolving {len(location_names)} locations")
            self.current_query_logger.info(f"🌍 Resolving {len(location_names)} locations: {location_names}")
            
            # Resolve each location
            resolved_locations = []
            for location_name in location_names:
                self.current_query_logger.info(f"🔍 Resolving location: '{location_name}'")
                location = await self._resolve_location(location_name)
                if location:
                    resolved_locations.append(location)
                    self.logger.info(f"Resolved '{location_name}' -> {location.name} (conf: {location.confidence:.2f})")
                    self.current_query_logger.info(f"✅ Resolved '{location_name}' -> {location.name} (confidence: {location.confidence:.2f})")
                    self.current_query_logger.info(f"   Bounds: N:{location.bounds.north}, S:{location.bounds.south}, E:{location.bounds.east}, W:{location.bounds.west}")
                else:
                    self.logger.warning(f"Could not resolve location: {location_name}")
                    self.current_query_logger.warning(f"❌ Could not resolve location: {location_name}")
            
            self.current_query_logger.info(f"✅ Successfully resolved {len(resolved_locations)}/{len(location_names)} locations")
            
            # Generate spatial query
            spatial_query = self._generate_spatial_query(resolved_locations, context or {})
            
            result = AgentResult.success_result(
                self.agent_name,
                {
                    'locations': [self._location_to_dict(loc) for loc in resolved_locations],
                    'spatial_query': self._spatial_query_to_dict(spatial_query),
                    'unresolved_locations': [name for name in location_names 
                                           if not any(loc.name.lower() == name.lower() or 
                                                    name.lower() in [alias.lower() for alias in (loc.aliases or [])]
                                                    for loc in resolved_locations)]
                },
                {'total_locations': len(resolved_locations)}
            )
            
            # Log successful result
            self.current_query_logger.log_result(self.agent_name, result)
            return result
            
        except Exception as e:
            # Log error with full context
            self.current_query_logger.log_error(
                self.agent_name, e, "Location processing"
            )
            self.logger.error(f"Error processing locations: {str(e)}")
            return AgentResult.error_result(
                self.agent_name,
                [f"Failed to process locations: {str(e)}"]
            )
        finally:
            # Don't close shared query logger - it will be closed by orchestrator
            self.current_query_logger = None
    
    async def _resolve_location(self, location_name: str) -> Optional[Location]:
        """Resolve a single location name to coordinates"""
        
        location_lower = location_name.lower().strip()
        
        # First check predefined locations
        if location_lower in self.predefined_locations:
            return self.predefined_locations[location_lower]
        
        # Check aliases
        for key, location in self.predefined_locations.items():
            if location.aliases and any(alias.lower() == location_lower for alias in location.aliases):
                return location
        
        # Try to parse coordinates
        coord_location = self._parse_coordinates(location_name)
        if coord_location:
            return coord_location
        
        # Use LLM for unknown locations
        return await self._resolve_with_llm(location_name)
    
    def _parse_coordinates(self, location_str: str) -> Optional[Location]:
        """Parse coordinate strings like '40.7,-74.0' or '40°N 74°W'"""
        
        # Pattern for decimal coordinates: lat,lon
        decimal_pattern = r'^(-?\d+\.?\d*),\s*(-?\d+\.?\d*)$'
        match = re.match(decimal_pattern, location_str.strip())
        
        if match:
            lat, lon = float(match.group(1)), float(match.group(2))
            
            # Validate coordinate ranges
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                # Create small bounding box around point
                buffer = 0.1  # 0.1 degree buffer
                return Location(
                    name=f"Point ({lat:.3f}, {lon:.3f})",
                    location_type=LocationType.COORDINATES,
                    bounds=GeographicBounds(
                        north=lat + buffer,
                        south=lat - buffer,
                        east=lon + buffer,
                        west=lon - buffer
                    ),
                    center_lat=lat,
                    center_lon=lon,
                    confidence=0.99
                )
        
        # Pattern for DMS coordinates: 40°N 74°W
        dms_pattern = r'(\d+)°([NS])\s+(\d+)°([EW])'
        match = re.match(dms_pattern, location_str.strip(), re.IGNORECASE)
        
        if match:
            lat_deg, lat_dir, lon_deg, lon_dir = match.groups()
            
            lat = float(lat_deg) * (1 if lat_dir.upper() == 'N' else -1)
            lon = float(lon_deg) * (1 if lon_dir.upper() == 'E' else -1)
            
            buffer = 0.1
            return Location(
                name=f"Point ({lat:.1f}°{lat_dir.upper()}, {lon:.1f}°{lon_dir.upper()})",
                location_type=LocationType.COORDINATES,
                bounds=GeographicBounds(
                    north=lat + buffer,
                    south=lat - buffer,
                    east=lon + buffer,
                    west=lon - buffer
                ),
                center_lat=lat,
                center_lon=lon,
                confidence=0.99
            )
        
        return None
    
    async def _resolve_with_llm(self, location_name: str) -> Optional[Location]:
        """Use LLM to resolve unknown location names"""
        
        system_prompt = """You are a geographic location resolver specializing in oceanographic regions.
Given a location name, return the geographic bounds and information.

Return a JSON object with this structure:
{
    "name": "standardized location name",
    "type": "ocean|sea|gulf|bay|strait|unknown",
    "bounds": {
        "north": latitude,
        "south": latitude,
        "east": longitude,
        "west": longitude
    },
    "center_lat": latitude,
    "center_lon": longitude,
    "confidence": 0.0-1.0,
    "aliases": ["alternative names"]
}

If the location cannot be resolved, return null.
Focus on marine/oceanic regions. Use standard geographic boundaries."""
        
        user_prompt = f"Resolve this marine location: \"{location_name}\""
        
        messages = [
            self.create_system_message(system_prompt),
            self.create_user_message(user_prompt)
        ]
        
        try:
            response = await self.call_llm(messages, temperature=0.1)
            
            # Extract and parse JSON
            extracted_json = extract_json_string(response.content)
            if extracted_json is None:
                self.logger.warning("No valid JSON found in LLM response")
                return None
                
            location_data = json.loads(extracted_json)
            
            if location_data is None:
                return None
            
            return Location(
                name=location_data['name'],
                location_type=LocationType(location_data['type']),
                bounds=GeographicBounds(
                    north=location_data['bounds']['north'],
                    south=location_data['bounds']['south'],
                    east=location_data['bounds']['east'],
                    west=location_data['bounds']['west']
                ),
                center_lat=location_data['center_lat'],
                center_lon=location_data['center_lon'],
                confidence=location_data['confidence'],
                aliases=location_data.get('aliases', [])
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.warning(f"Failed to resolve location '{location_name}' with LLM: {e}")
            return None
    
    def _generate_spatial_query(self, locations: List[Location], context: Dict[str, Any]) -> SpatialQuery:
        """Generate spatial query information from resolved locations"""
        
        # Determine spatial operation based on context
        spatial_operation = "intersects"  # Default
        if len(locations) > 1:
            if context.get('comparison_intent') == 'spatial':
                spatial_operation = "comparison"
            else:
                spatial_operation = "union"
        
        # Determine resolution based on area coverage
        total_area = 0
        for location in locations:
            lat_span = location.bounds.north - location.bounds.south
            lon_span = location.bounds.east - location.bounds.west
            total_area += lat_span * lon_span
        
        if total_area > 10000:  # Very large area
            resolution = "coarse"
        elif total_area > 1000:  # Medium area
            resolution = "medium"
        else:
            resolution = "fine"
        
        return SpatialQuery(
            locations=locations,
            spatial_operation=spatial_operation,
            resolution=resolution,
            projection=None  # Will be determined by visualization agent
        )
    
    def _location_to_dict(self, location: Location) -> Dict[str, Any]:
        """Convert Location object to dictionary"""
        return {
            'name': location.name,
            'type': location.location_type.value,
            'bounds': location.bounds.to_dict(),
            'center_lat': location.center_lat,
            'center_lon': location.center_lon,
            'confidence': location.confidence,
            'aliases': location.aliases or []
        }
    
    def _spatial_query_to_dict(self, spatial_query: SpatialQuery) -> Dict[str, Any]:
        """Convert SpatialQuery object to dictionary"""
        return {
            'spatial_operation': spatial_query.spatial_operation,
            'resolution': spatial_query.resolution,
            'projection': spatial_query.projection,
            'location_count': len(spatial_query.locations)
        }
    
    def get_bounding_box_union(self, locations: List[Location]) -> GeographicBounds:
        """Get the union of all location bounding boxes"""
        if not locations:
            return GeographicBounds(north=90, south=-90, east=180, west=-180)
        
        north = max(loc.bounds.north for loc in locations)
        south = min(loc.bounds.south for loc in locations)
        east = max(loc.bounds.east for loc in locations)
        west = min(loc.bounds.west for loc in locations)
        
        return GeographicBounds(north=north, south=south, east=east, west=west)
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return information about geospatial capabilities"""
        return {
            "description": "Resolves location names to geographic coordinates and bounding boxes",
            "predefined_locations": list(self.predefined_locations.keys()),
            "supported_formats": ["location names", "decimal coordinates", "DMS coordinates"],
            "coordinate_system": "WGS84",
            "location_types": [loc_type.value for loc_type in LocationType]
        }