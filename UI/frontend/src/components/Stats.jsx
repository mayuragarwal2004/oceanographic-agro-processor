import { useState, useEffect } from 'react'
import axios from 'axios'

const API_BASE_URL = 'http://localhost:3001/api'

function Stats() {
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchStats()
  }, [])

  const fetchStats = async () => {
    try {
      setLoading(true)
      const response = await axios.get(`${API_BASE_URL}/stats`)
      setStats(response.data)
      setError(null)
    } catch (err) {
      setError('Failed to fetch statistics')
      console.error('Error fetching stats:', err)
    } finally {
      setLoading(false)
    }
  }

  if (loading) return <div className="loading">Loading statistics...</div>
  if (error) return <div className="error">{error}</div>
  if (!stats) return <div className="error">No data available</div>

  const formatNumber = (num) => {
    if (num === null || num === undefined) return 'N/A'
    return new Intl.NumberFormat().format(Math.round(num))
  }

  const formatDate = (dateStr) => {
    if (!dateStr) return 'N/A'
    return new Date(dateStr).toLocaleDateString()
  }

  const formatDecimal = (num, decimals = 2) => {
    if (num === null || num === undefined) return 'N/A'
    return Number(num).toFixed(decimals)
  }

  return (
    <div className="stats-container">
      <div className="card">
        <h2 style={{ color: '#2c3e50', textAlign: 'center', marginBottom: '2rem' }}>
          📈 Oceanographic Database Statistics
        </h2>
        <p style={{ color: '#6c757d', textAlign: 'center', fontSize: '1.1rem', marginBottom: '2rem' }}>
          Real-time insights into your Argo float dataset
        </p>
      </div>
      
      <div className="stats-grid">
        <div className="stat-card">
          <h3 style={{ color: '#2c3e50' }}>📊 Dataset Overview</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem', textAlign: 'center' }}>
            <div>
              <div className="value" style={{ color: '#3498db' }}>{formatNumber(stats.profileCount?.count)}</div>
              <div className="label" style={{ color: '#5a6c7d' }}>Profiles</div>
            </div>
            <div>
              <div className="value" style={{ color: '#e74c3c' }}>{formatNumber(stats.measurementCount?.count)}</div>
              <div className="label" style={{ color: '#5a6c7d' }}>Measurements</div>
            </div>
            <div>
              <div className="value" style={{ color: '#27ae60' }}>{formatNumber(stats.platformCount?.count)}</div>
              <div className="label" style={{ color: '#5a6c7d' }}>Platforms</div>
            </div>
          </div>
        </div>

        <div className="stat-card">
          <h3 style={{ color: '#2c3e50' }}>📅 Temporal Coverage</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', textAlign: 'center' }}>
            <div>
              <div className="value" style={{ color: '#9b59b6', fontSize: '1.3rem' }}>{formatDate(stats.dateRange?.min_date)}</div>
              <div className="label" style={{ color: '#5a6c7d' }}>First Record</div>
            </div>
            <div>
              <div className="value" style={{ color: '#9b59b6', fontSize: '1.3rem' }}>{formatDate(stats.dateRange?.max_date)}</div>
              <div className="label" style={{ color: '#5a6c7d' }}>Latest Record</div>
            </div>
          </div>
        </div>

        <div className="stat-card">
          <h3 style={{ color: '#2c3e50' }}>🌡️ Temperature Analysis</h3>
          <div style={{ textAlign: 'center', marginBottom: '1rem' }}>
            <div className="value" style={{ color: '#e74c3c' }}>{formatDecimal(stats.avgTemperature?.avg_temp)}°C</div>
            <div className="label" style={{ color: '#5a6c7d' }}>Global Average</div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', textAlign: 'center', fontSize: '0.9rem' }}>
            <div>
              <strong style={{ color: '#3498db' }}>{formatDecimal(stats.avgTemperature?.min_temp)}°C</strong>
              <div style={{ color: '#7f8c8d' }}>Minimum</div>
            </div>
            <div>
              <strong style={{ color: '#e67e22' }}>{formatDecimal(stats.avgTemperature?.max_temp)}°C</strong>
              <div style={{ color: '#7f8c8d' }}>Maximum</div>
            </div>
          </div>
        </div>

        <div className="stat-card">
          <h3 style={{ color: '#2c3e50' }}>🧂 Salinity Analysis</h3>
          <div style={{ textAlign: 'center', marginBottom: '1rem' }}>
            <div className="value" style={{ color: '#3498db' }}>{formatDecimal(stats.avgSalinity?.avg_salinity)} PSU</div>
            <div className="label" style={{ color: '#5a6c7d' }}>Global Average</div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', textAlign: 'center', fontSize: '0.9rem' }}>
            <div>
              <strong style={{ color: '#27ae60' }}>{formatDecimal(stats.avgSalinity?.min_salinity)} PSU</strong>
              <div style={{ color: '#7f8c8d' }}>Minimum</div>
            </div>
            <div>
              <strong style={{ color: '#8e44ad' }}>{formatDecimal(stats.avgSalinity?.max_salinity)} PSU</strong>
              <div style={{ color: '#7f8c8d' }}>Maximum</div>
            </div>
          </div>
        </div>
      </div>

      <div className="card">
        <h3 style={{ color: '#2c3e50', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          🌊 About This Oceanographic Dataset
        </h3>
        <p style={{ color: '#5a6c7d', fontSize: '1.05rem', lineHeight: '1.6', marginBottom: '1.5rem' }}>
          This database contains comprehensive measurements from <strong style={{ color: '#3498db' }}>Argo floats</strong> - 
          autonomous profiling instruments that drift in the ocean and periodically surface to transmit 
          temperature, salinity, and pressure data. This global network provides critical insights into 
          ocean conditions and climate patterns.
        </p>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1.5rem' }}>
          <div style={{ padding: '1rem', background: '#f8f9fa', borderRadius: '8px', borderLeft: '4px solid #3498db' }}>
            <h4 style={{ color: '#2c3e50', margin: '0 0 0.5rem 0', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              📊 Profiles
            </h4>
            <p style={{ color: '#5a6c7d', margin: 0, fontSize: '0.95rem' }}>
              Individual measurement cycles from each float, representing one complete ascent or descent through the water column
            </p>
          </div>
          
          <div style={{ padding: '1rem', background: '#f8f9fa', borderRadius: '8px', borderLeft: '4px solid #e74c3c' }}>
            <h4 style={{ color: '#2c3e50', margin: '0 0 0.5rem 0', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              🌡️ Measurements
            </h4>
            <p style={{ color: '#5a6c7d', margin: 0, fontSize: '0.95rem' }}>
              Temperature, salinity, and pressure readings collected at specific depths during each profile
            </p>
          </div>
          
          <div style={{ padding: '1rem', background: '#f8f9fa', borderRadius: '8px', borderLeft: '4px solid #27ae60' }}>
            <h4 style={{ color: '#2c3e50', margin: '0 0 0.5rem 0', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              🚀 Platforms
            </h4>
            <p style={{ color: '#5a6c7d', margin: 0, fontSize: '0.95rem' }}>
              Individual Argo float devices, each uniquely identified and contributing to the global ocean monitoring network
            </p>
          </div>
        </div>
        
        <div style={{ marginTop: '1.5rem', padding: '1rem', background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', borderRadius: '8px', color: 'white', textAlign: 'center' }}>
          <strong>🔗 Data Relationships:</strong> Each Platform operates multiple Profiles over time, and each Profile contains multiple Measurements at different depths
        </div>
      </div>
    </div>
  )
}

export default Stats