const express = require('express');
const cors = require('cors');
const { Pool } = require('pg');
require('dotenv').config({ path: '../../.env' });

const app = express();
const port = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());

// Database connection
const pool = new Pool({
  host: process.env.DB_HOST,
  port: process.env.DB_PORT,
  database: process.env.DB_NAME,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
});

// Test database connection
pool.connect((err) => {
  if (err) {
    console.error('Database connection error:', err);
  } else {
    console.log('Connected to PostgreSQL database');
  }
});

// Routes

// Get all profiles with platform info (limited to 50 for performance)
app.get('/api/profiles', async (req, res) => {
  try {
    const limit = req.query.limit || 50;
    const query = `
      SELECT 
        p.id,
        p.cycle_number,
        p.profile_date,
        p.latitude,
        p.longitude,
        p.direction,
        p.data_mode,
        pl.platform_number,
        pl.project_name,
        pl.institution
      FROM profiles p
      JOIN platforms pl ON p.platform_id = pl.id
      ORDER BY p.profile_date DESC
      LIMIT $1
    `;
    
    const result = await pool.query(query, [limit]);
    res.json(result.rows);
  } catch (err) {
    console.error('Error fetching profiles:', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get measurements for a specific profile
app.get('/api/profiles/:profileId/measurements', async (req, res) => {
  try {
    const { profileId } = req.params;
    const query = `
      SELECT 
        id,
        pressure,
        temp as temperature,
        psal as salinity,
        pressure_qc,
        temp_qc as temperature_qc,
        psal_qc as salinity_qc,
        pressure_adjusted,
        temp_adjusted as temperature_adjusted,
        psal_adjusted as salinity_adjusted,
        pressure_adjusted_qc,
        temp_adjusted_qc as temperature_adjusted_qc,
        psal_adjusted_qc as salinity_adjusted_qc,
        pressure_adjusted_error,
        temp_adjusted_error as temperature_adjusted_error,
        psal_adjusted_error as salinity_adjusted_error
      FROM measurements
      WHERE profile_id = $1
      ORDER BY pressure ASC
    `;
    
    const result = await pool.query(query, [profileId]);
    res.json(result.rows);
  } catch (err) {
    console.error('Error fetching measurements:', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get summary statistics
app.get('/api/stats', async (req, res) => {
  try {
    const queries = {
      profileCount: 'SELECT COUNT(*) as count FROM profiles',
      measurementCount: 'SELECT COUNT(*) as count FROM measurements',
      platformCount: 'SELECT COUNT(*) as count FROM platforms',
      dateRange: `
        SELECT 
          MIN(profile_date) as min_date, 
          MAX(profile_date) as max_date 
        FROM profiles
      `,
      avgTemperature: `
        SELECT 
          AVG(temp) as avg_temp,
          MIN(temp) as min_temp,
          MAX(temp) as max_temp
        FROM measurements 
        WHERE temp IS NOT NULL
      `,
      avgSalinity: `
        SELECT 
          AVG(psal) as avg_salinity,
          MIN(psal) as min_salinity,
          MAX(psal) as max_salinity
        FROM measurements 
        WHERE psal IS NOT NULL
      `
    };

    const results = {};
    for (const [key, query] of Object.entries(queries)) {
      const result = await pool.query(query);
      results[key] = result.rows[0];
    }

    res.json(results);
  } catch (err) {
    console.error('Error fetching stats:', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get temperature and salinity data for charts (sample data)
app.get('/api/chart-data', async (req, res) => {
  try {
    const query = `
      SELECT 
        p.profile_date,
        AVG(m.temp) as avg_temperature,
        AVG(m.psal) as avg_salinity,
        p.latitude,
        p.longitude
      FROM profiles p
      JOIN measurements m ON p.id = m.profile_id
      WHERE m.temp IS NOT NULL AND m.psal IS NOT NULL
      GROUP BY p.id, p.profile_date, p.latitude, p.longitude
      ORDER BY p.profile_date
      LIMIT 100
    `;
    
    const result = await pool.query(query);
    res.json(result.rows);
  } catch (err) {
    console.error('Error fetching chart data:', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get profile details by ID
app.get('/api/profiles/:profileId', async (req, res) => {
  try {
    const { profileId } = req.params;
    const query = `
      SELECT 
        p.*,
        pl.platform_number,
        pl.project_name,
        pl.institution,
        pl.pi_name
      FROM profiles p
      JOIN platforms pl ON p.platform_id = pl.id
      WHERE p.id = $1
    `;
    
    const result = await pool.query(query, [profileId]);
    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'Profile not found' });
    }

    res.json(result.rows[0]);
  } catch (err) {
    console.error('Error fetching profile details:', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get all profile locations for map display
app.get('/api/profile-locations', async (req, res) => {
  try {
    const limit = req.query.limit || 200;
    const query = `
      SELECT 
        p.id,
        p.latitude,
        p.longitude,
        p.profile_date,
        p.cycle_number,
        pl.platform_number,
        pl.institution
      FROM profiles p
      JOIN platforms pl ON p.platform_id = pl.id
      WHERE p.latitude IS NOT NULL AND p.longitude IS NOT NULL
      ORDER BY p.profile_date DESC
      LIMIT $1
    `;
    
    const result = await pool.query(query, [limit]);
    res.json(result.rows);
  } catch (err) {
    console.error('Error fetching profile locations:', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});