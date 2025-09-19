-- PostGIS Helper Functions for Ocean Data Analysis
-- Run these SQL statements after running Prisma migrate to enable advanced spatial queries

-- Function to create a PostGIS point from latitude and longitude
CREATE OR REPLACE FUNCTION create_point_from_lat_lon(lat FLOAT, lon FLOAT)
RETURNS geometry(Point, 4326) AS $$
BEGIN
  RETURN ST_SetSRID(ST_MakePoint(lon, lat), 4326);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to find profiles within a bounding box
CREATE OR REPLACE FUNCTION profiles_in_bbox(
  min_lat FLOAT, 
  min_lon FLOAT, 
  max_lat FLOAT, 
  max_lon FLOAT
)
RETURNS TABLE(profile_id TEXT) AS $$
BEGIN
  RETURN QUERY
  SELECT p.id
  FROM profiles p
  WHERE p.location && ST_MakeEnvelope(min_lon, min_lat, max_lon, max_lat, 4326);
END;
$$ LANGUAGE plpgsql;

-- Function to find profiles within a radius (in kilometers) of a point
CREATE OR REPLACE FUNCTION profiles_within_radius(
  center_lat FLOAT, 
  center_lon FLOAT, 
  radius_km FLOAT
)
RETURNS TABLE(
  profile_id TEXT,
  distance_km FLOAT
) AS $$
BEGIN
  RETURN QUERY
  SELECT 
    p.id,
    ST_Distance(
      p.location::geography,
      ST_SetSRID(ST_MakePoint(center_lon, center_lat), 4326)::geography
    ) / 1000.0 AS distance_km
  FROM profiles p
  WHERE ST_DWithin(
    p.location::geography,
    ST_SetSRID(ST_MakePoint(center_lon, center_lat), 4326)::geography,
    radius_km * 1000
  )
  ORDER BY distance_km;
END;
$$ LANGUAGE plpgsql;

-- Function to get ocean region for a profile based on its location
CREATE OR REPLACE FUNCTION get_profile_region(profile_lat FLOAT, profile_lon FLOAT)
RETURNS TEXT AS $$
DECLARE
  region_name TEXT;
BEGIN
  SELECT r.name INTO region_name
  FROM ocean_regions r
  WHERE ST_Contains(r.boundary, ST_SetSRID(ST_MakePoint(profile_lon, profile_lat), 4326))
  LIMIT 1;
  
  RETURN COALESCE(region_name, 'Unknown');
END;
$$ LANGUAGE plpgsql;

-- Function to create a convex hull around all profiles from a specific platform
CREATE OR REPLACE FUNCTION platform_coverage_area(platform_num TEXT)
RETURNS geometry(Polygon, 4326) AS $$
DECLARE
  coverage_area geometry(Polygon, 4326);
BEGIN
  SELECT ST_ConvexHull(ST_Collect(p.location))
  INTO coverage_area
  FROM profiles p
  JOIN platforms pl ON p.platform_id = pl.id
  WHERE pl.platform_number = platform_num;
  
  RETURN coverage_area;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate the average position of all profiles in a time range
CREATE OR REPLACE FUNCTION average_position_in_period(
  start_date TIMESTAMP,
  end_date TIMESTAMP
)
RETURNS geometry(Point, 4326) AS $$
BEGIN
  RETURN (
    SELECT ST_Centroid(ST_Collect(p.location))
    FROM profiles p
    WHERE p.profile_date BETWEEN start_date AND end_date
  );
END;
$$ LANGUAGE plpgsql;

-- Trigger function to automatically populate the location field when lat/lon are inserted/updated
CREATE OR REPLACE FUNCTION update_profile_location()
RETURNS TRIGGER AS $$
BEGIN
  NEW.location = ST_SetSRID(ST_MakePoint(NEW.longitude, NEW.latitude), 4326);
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create the trigger (uncomment after running Prisma migrate)
-- CREATE TRIGGER update_profile_location_trigger
--   BEFORE INSERT OR UPDATE OF latitude, longitude
--   ON profiles
--   FOR EACH ROW
--   EXECUTE FUNCTION update_profile_location();

-- Useful indexes for common oceanographic queries
-- CREATE INDEX idx_profiles_date_location ON profiles USING GIST(profile_date, location);
-- CREATE INDEX idx_profiles_platform_date ON profiles(platform_id, profile_date);
-- CREATE INDEX idx_measurements_temp_depth ON measurements(temperature, pressure) WHERE temperature IS NOT NULL;
-- CREATE INDEX idx_measurements_salinity_depth ON measurements(salinity, pressure) WHERE salinity IS NOT NULL;

-- Example queries you can use in your application:

-- 1. Find all profiles in the Indian Ocean (example bounding box)
/*
SELECT p.id, p.latitude, p.longitude, p.profile_date
FROM profiles p
WHERE p.location && ST_MakeEnvelope(30, -40, 120, 30, 4326);
*/

-- 2. Find profiles within 100km of a specific point
/*
SELECT * FROM profiles_within_radius(-12.5, 45.0, 100);
*/

-- 3. Get temperature measurements within a spatial region and depth range
/*
SELECT p.latitude, p.longitude, m.pressure, m.temperature
FROM profiles p
JOIN measurements m ON p.id = m.profile_id
WHERE p.location && ST_MakeEnvelope(70, -30, 90, -10, 4326)  -- Example: Southern Indian Ocean
  AND m.pressure BETWEEN 0 AND 100  -- Surface to 100m depth
  AND m.temperature IS NOT NULL;
*/

-- 4. Calculate average sea surface temperature (top 10m) by month in a region
/*
SELECT 
  EXTRACT(YEAR FROM p.profile_date) as year,
  EXTRACT(MONTH FROM p.profile_date) as month,
  AVG(m.temperature) as avg_sst,
  COUNT(*) as measurement_count
FROM profiles p
JOIN measurements m ON p.id = m.profile_id
WHERE p.location && ST_MakeEnvelope(-80, 20, -60, 40, 4326)  -- Example: Western Atlantic
  AND m.pressure <= 10  -- Top 10 meters
  AND m.temperature IS NOT NULL
GROUP BY EXTRACT(YEAR FROM p.profile_date), EXTRACT(MONTH FROM p.profile_date)
ORDER BY year, month;
*/