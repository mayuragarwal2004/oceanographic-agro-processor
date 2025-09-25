import { useState, useEffect } from 'react'
import axios from 'axios'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter } from 'recharts'

const API_BASE_URL = 'http://localhost:3001/api'

function Charts() {
  const [chartData, setChartData] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchChartData()
  }, [])

  const fetchChartData = async () => {
    try {
      setLoading(true)
      const response = await axios.get(`${API_BASE_URL}/chart-data`)
      const processedData = response.data.map(item => ({
        ...item,
        date: new Date(item.profile_date).toLocaleDateString(),
        shortDate: new Date(item.profile_date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
        avg_temperature: parseFloat(item.avg_temperature),
        avg_salinity: parseFloat(item.avg_salinity),
        latitude: parseFloat(item.latitude),
        longitude: parseFloat(item.longitude)
      }))
      setChartData(processedData)
      setError(null)
    } catch (err) {
      setError('Failed to fetch chart data')
      console.error('Error fetching chart data:', err)
    } finally {
      setLoading(false)
    }
  }

  if (loading) return <div className="loading">Loading charts...</div>
  if (error) return <div className="error">{error}</div>
  if (chartData.length === 0) return <div className="error">No data available for charts</div>

  return (
    <div className="charts-container">
      <div className="card">
        <h2 style={{ color: '#2c3e50', textAlign: 'center', marginBottom: '1rem' }}>
          📈 Oceanographic Data Visualizations
        </h2>
        <p style={{ color: '#6c757d', textAlign: 'center', marginBottom: '2rem' }}>
          Interactive charts showing temperature, salinity patterns and relationships
        </p>
      </div>
      
      {/* First row - Time series charts */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginBottom: '2rem' }}>
        <div className="chart-card">
          <h3 style={{ color: '#2c3e50', textAlign: 'center', marginBottom: '1rem' }}>
            🌡️ Temperature Over Time
          </h3>
          <ResponsiveContainer width="100%" height={350}>
            <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
              <XAxis 
                dataKey="shortDate" 
                tick={{ fontSize: 11, fill: '#2c3e50' }}
                interval="preserveStartEnd"
                axisLine={{ stroke: '#2c3e50' }}
                tickLine={{ stroke: '#2c3e50' }}
              />
              <YAxis 
                label={{ 
                  value: 'Temperature (°C)', 
                  angle: -90, 
                  position: 'insideLeft',
                  style: { textAnchor: 'middle', fill: '#2c3e50' }
                }}
                tick={{ fontSize: 11, fill: '#2c3e50' }}
                axisLine={{ stroke: '#2c3e50' }}
                tickLine={{ stroke: '#2c3e50' }}
              />
              <Tooltip 
                formatter={(value, name) => [
                  `${parseFloat(value).toFixed(2)}°C`, 
                  'Average Temperature'
                ]}
                labelFormatter={(label) => `Date: ${label}`}
                contentStyle={{ 
                  backgroundColor: 'white', 
                  border: '1px solid #ccc',
                  borderRadius: '8px',
                  color: '#2c3e50'
                }}
              />
              <Legend wrapperStyle={{ color: '#2c3e50' }} />
              <Line 
                type="monotone" 
                dataKey="avg_temperature" 
                stroke="#e74c3c" 
                strokeWidth={3}
                name="Average Temperature"
                dot={{ fill: '#e74c3c', strokeWidth: 2, r: 5 }}
                activeDot={{ r: 8, fill: '#c0392b' }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <h3 style={{ color: '#2c3e50', textAlign: 'center', marginBottom: '1rem' }}>
            🧂 Salinity Over Time
          </h3>
          <ResponsiveContainer width="100%" height={350}>
            <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
              <XAxis 
                dataKey="shortDate" 
                tick={{ fontSize: 11, fill: '#2c3e50' }}
                interval="preserveStartEnd"
                axisLine={{ stroke: '#2c3e50' }}
                tickLine={{ stroke: '#2c3e50' }}
              />
              <YAxis 
                label={{ 
                  value: 'Salinity (PSU)', 
                  angle: -90, 
                  position: 'insideLeft',
                  style: { textAnchor: 'middle', fill: '#2c3e50' }
                }}
                tick={{ fontSize: 11, fill: '#2c3e50' }}
                axisLine={{ stroke: '#2c3e50' }}
                tickLine={{ stroke: '#2c3e50' }}
              />
              <Tooltip 
                formatter={(value, name) => [
                  `${parseFloat(value).toFixed(3)} PSU`, 
                  'Average Salinity'
                ]}
                labelFormatter={(label) => `Date: ${label}`}
                contentStyle={{ 
                  backgroundColor: 'white', 
                  border: '1px solid #ccc',
                  borderRadius: '8px',
                  color: '#2c3e50'
                }}
              />
              <Legend wrapperStyle={{ color: '#2c3e50' }} />
              <Line 
                type="monotone" 
                dataKey="avg_salinity" 
                stroke="#3498db" 
                strokeWidth={3}
                name="Average Salinity"
                dot={{ fill: '#3498db', strokeWidth: 2, r: 5 }}
                activeDot={{ r: 8, fill: '#2980b9' }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Second row - Relationship charts */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginBottom: '2rem' }}>
        <div className="chart-card">
          <h3 style={{ color: '#2c3e50', textAlign: 'center', marginBottom: '1rem' }}>
            🌊 Temperature vs Salinity Relationship
          </h3>
          <ResponsiveContainer width="100%" height={350}>
            <ScatterChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 40 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
              <XAxis 
                dataKey="avg_temperature" 
                type="number"
                domain={['dataMin - 1', 'dataMax + 1']}
                label={{ 
                  value: 'Temperature (°C)', 
                  position: 'insideBottom', 
                  offset: -5,
                  style: { textAnchor: 'middle', fill: '#2c3e50' }
                }}
                tick={{ fontSize: 11, fill: '#2c3e50' }}
                axisLine={{ stroke: '#2c3e50' }}
                tickLine={{ stroke: '#2c3e50' }}
              />
              <YAxis 
                dataKey="avg_salinity" 
                type="number"
                domain={['dataMin - 0.1', 'dataMax + 0.1']}
                label={{ 
                  value: 'Salinity (PSU)', 
                  angle: -90, 
                  position: 'insideLeft',
                  style: { textAnchor: 'middle', fill: '#2c3e50' }
                }}
                tick={{ fontSize: 11, fill: '#2c3e50' }}
                axisLine={{ stroke: '#2c3e50' }}
                tickLine={{ stroke: '#2c3e50' }}
              />
              <Tooltip 
                formatter={(value, name, props) => {
                  const temp = props.payload.avg_temperature
                  const sal = props.payload.avg_salinity
                  return [
                    `Temp: ${temp?.toFixed(2)}°C, Sal: ${sal?.toFixed(3)} PSU`,
                    'T-S Relationship'
                  ]
                }}
                labelFormatter={() => 'Temperature-Salinity Plot'}
                contentStyle={{ 
                  backgroundColor: 'white', 
                  border: '1px solid #ccc',
                  borderRadius: '8px',
                  color: '#2c3e50'
                }}
              />
              <Scatter 
                name="T-S Relationship" 
                fill="#9b59b6"
                fillOpacity={0.8}
                stroke="#8e44ad"
                strokeWidth={2}
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <h3 style={{ color: '#2c3e50', textAlign: 'center', marginBottom: '1rem' }}>
            🗺️ Geographic Distribution
          </h3>
          <ResponsiveContainer width="100%" height={350}>
            <ScatterChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 40 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
              <XAxis 
                dataKey="longitude" 
                type="number"
                domain={['dataMin - 5', 'dataMax + 5']}
                label={{ 
                  value: 'Longitude (°)', 
                  position: 'insideBottom', 
                  offset: -5,
                  style: { textAnchor: 'middle', fill: '#2c3e50' }
                }}
                tick={{ fontSize: 11, fill: '#2c3e50' }}
                axisLine={{ stroke: '#2c3e50' }}
                tickLine={{ stroke: '#2c3e50' }}
              />
              <YAxis 
                dataKey="latitude" 
                type="number"
                domain={['dataMin - 5', 'dataMax + 5']}
                label={{ 
                  value: 'Latitude (°)', 
                  angle: -90, 
                  position: 'insideLeft',
                  style: { textAnchor: 'middle', fill: '#2c3e50' }
                }}
                tick={{ fontSize: 11, fill: '#2c3e50' }}
                axisLine={{ stroke: '#2c3e50' }}
                tickLine={{ stroke: '#2c3e50' }}
              />
              <Tooltip 
                formatter={(value, name, props) => {
                  const lat = props.payload.latitude
                  const lon = props.payload.longitude
                  const temp = props.payload.avg_temperature
                  const date = props.payload.date
                  return [
                    `${lat?.toFixed(4)}°, ${lon?.toFixed(4)}° - ${temp?.toFixed(2)}°C on ${date}`,
                    'Location & Temperature'
                  ]
                }}
                labelFormatter={() => 'Geographic Distribution'}
                contentStyle={{ 
                  backgroundColor: 'white', 
                  border: '1px solid #ccc',
                  borderRadius: '8px',
                  color: '#2c3e50'
                }}
              />
              <Scatter 
                name="Measurement Locations" 
                fill="#2ecc71"
                fillOpacity={0.8}
                stroke="#27ae60"
                strokeWidth={2}
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card">
        <h3 style={{ color: '#2c3e50', textAlign: 'center', marginBottom: '1.5rem' }}>
          📊 Chart Information & Analysis
        </h3>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginBottom: '1.5rem' }}>
          <div>
            <div style={{ padding: '1rem', background: '#f8f9fa', borderRadius: '8px', marginBottom: '1rem', borderLeft: '4px solid #e74c3c' }}>
              <h4 style={{ color: '#2c3e50', margin: '0 0 0.5rem 0' }}>🌡️ Temperature Time Series</h4>
              <p style={{ color: '#5a6c7d', margin: 0, fontSize: '0.95rem' }}>
                Shows how average temperature varies over time across all profiles. Helps identify seasonal patterns and trends.
              </p>
            </div>
            
            <div style={{ padding: '1rem', background: '#f8f9fa', borderRadius: '8px', borderLeft: '4px solid #3498db' }}>
              <h4 style={{ color: '#2c3e50', margin: '0 0 0.5rem 0' }}>🧂 Salinity Time Series</h4>
              <p style={{ color: '#5a6c7d', margin: 0, fontSize: '0.95rem' }}>
                Displays temporal variation in average salinity measurements. Critical for understanding ocean circulation patterns.
              </p>
            </div>
          </div>
          <div>
            <div style={{ padding: '1rem', background: '#f8f9fa', borderRadius: '8px', marginBottom: '1rem', borderLeft: '4px solid #9b59b6' }}>
              <h4 style={{ color: '#2c3e50', margin: '0 0 0.5rem 0' }}>🌊 T-S Diagram</h4>
              <p style={{ color: '#5a6c7d', margin: 0, fontSize: '0.95rem' }}>
                Temperature-Salinity relationship helps identify different water masses and ocean processes. Each point represents one profile.
              </p>
            </div>
            
            <div style={{ padding: '1rem', background: '#f8f9fa', borderRadius: '8px', borderLeft: '4px solid #2ecc71' }}>
              <h4 style={{ color: '#2c3e50', margin: '0 0 0.5rem 0' }}>🗺️ Geographic Distribution</h4>
              <p style={{ color: '#5a6c7d', margin: 0, fontSize: '0.95rem' }}>
                Spatial distribution of measurement locations showing global data coverage and sampling density.
              </p>
            </div>
          </div>
        </div>
        <div style={{ 
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 
          color: 'white', 
          padding: '1rem', 
          borderRadius: '8px',
          textAlign: 'center'
        }}>
          <strong>📈 Data Note:</strong> Charts display data from up to 100 most recent profiles for optimal performance. 
          Each point represents the average of all measurements within a single profile cycle.
        </div>
      </div>
    </div>
  )
}

export default Charts