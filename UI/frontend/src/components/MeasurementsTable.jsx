import { useState, useEffect } from 'react'
import axios from 'axios'
import OpenStreetMap from './OpenStreetMap'

const API_BASE_URL = 'http://localhost:3001/api'

function MeasurementsTable({ profile, onBack }) {
  const [measurements, setMeasurements] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (profile) {
      fetchMeasurements()
    }
  }, [profile])

  const fetchMeasurements = async () => {
    try {
      setLoading(true)
      const response = await axios.get(`${API_BASE_URL}/profiles/${profile.id}/measurements`)
      setMeasurements(response.data)
      setError(null)
    } catch (err) {
      setError('Failed to fetch measurements')
      console.error('Error fetching measurements:', err)
    } finally {
      setLoading(false)
    }
  }

  const formatValue = (value, decimals = 3) => {
    if (value === null || value === undefined) return 'N/A'
    return Number(value).toFixed(decimals)
  }

  const getQcColor = (qc) => {
    if (qc === null || qc === undefined) return '#bdc3c7'
    switch (qc) {
      case 1: return '#27ae60' // Good
      case 2: return '#f39c12' // Probably good
      case 3: return '#e67e22' // Bad but correctable
      case 4: return '#e74c3c' // Bad
      case 5: return '#8e44ad' // Value changed
      case 8: return '#95a5a6' // Estimated
      case 9: return '#34495e' // Missing
      default: return '#bdc3c7'
    }
  }

  const getQcLabel = (qc) => {
    if (qc === null || qc === undefined) return 'N/A'
    switch (qc) {
      case 1: return 'Good'
      case 2: return 'Probably good'
      case 3: return 'Bad but correctable'
      case 4: return 'Bad'
      case 5: return 'Value changed'
      case 8: return 'Estimated'
      case 9: return 'Missing'
      default: return `QC ${qc}`
    }
  }

  if (loading) return <div className="loading">Loading measurements...</div>
  if (error) return <div className="error">{error}</div>

  return (
    <div className="measurements-container">
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem' }}>
        <button className="back-button" onClick={onBack}>
          ← Back to Profiles
        </button>
        <div style={{ color: '#6c757d' }}>
          Navigate back to see all profiles and select another one
        </div>
      </div>

      <div className="profile-header">
        <h2>🌊 Detailed Measurements for Profile</h2>
        <div className="profile-info">
          <div><strong>🏷️ Platform:</strong> <span style={{ color: '#f8f9fa', fontSize: '1.1rem', fontWeight: 'bold' }}>{profile.platform_number}</span></div>
          <div><strong>🔢 Cycle:</strong> <span style={{ color: '#f8f9fa' }}>{profile.cycle_number}</span></div>
          <div><strong>📅 Date:</strong> <span style={{ color: '#f8f9fa' }}>{new Date(profile.profile_date).toLocaleDateString()}</span></div>
          <div><strong>📍 Location:</strong> <span style={{ color: '#f8f9fa' }}>{Number(profile.latitude).toFixed(4)}°N, {Number(profile.longitude).toFixed(4)}°E</span></div>
          <div><strong>🏛️ Institution:</strong> <span style={{ color: '#f8f9fa' }}>{profile.institution || 'N/A'}</span></div>
          <div><strong>🔬 Project:</strong> <span style={{ color: '#f8f9fa' }}>{profile.project_name || 'N/A'}</span></div>
        </div>
      </div>

      {/* Location Map */}
      <div className="card">
        <h3 style={{ color: '#2c3e50', margin: '0 0 1rem 0', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          🗺️ Profile Location
          <span style={{ 
            fontSize: '0.8rem', 
            background: '#e8f4fd', 
            color: '#2980b9', 
            padding: '0.25rem 0.5rem', 
            borderRadius: '12px' 
          }}>
            {Number(profile.latitude).toFixed(4)}°N, {Number(profile.longitude).toFixed(4)}°E
          </span>
        </h3>
        <OpenStreetMap 
          locations={[profile]} 
          height="200px"
          zoom={4}
          center={[profile.latitude, profile.longitude]}
        />
      </div>

      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <h3 style={{ margin: 0, color: '#2c3e50' }}>📊 Measurement Data</h3>
            <p style={{ margin: '0.5rem 0', color: '#6c757d' }}>Temperature, salinity, and pressure readings with both actual and adjusted values</p>
          </div>
          <div className="measurement-count-badge" style={{ fontSize: '1rem', padding: '0.5rem 1rem' }}>
            {measurements.length} measurements
          </div>
        </div>

        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th colSpan="2" style={{ textAlign: 'center', borderBottom: '1px solid #ddd' }}>🌊 Pressure (dbar)</th>
                <th colSpan="2" style={{ textAlign: 'center', borderBottom: '1px solid #ddd' }}>🌡️ Temperature (°C)</th>
                <th colSpan="2" style={{ textAlign: 'center', borderBottom: '1px solid #ddd' }}>🧂 Salinity (PSU)</th>
                <th rowSpan="2" style={{ verticalAlign: 'middle' }}>📊 Adjustment Errors</th>
              </tr>
              <tr>
                <th style={{ fontSize: '0.85rem', color: '#3498db' }}>Actual</th>
                <th style={{ fontSize: '0.85rem', color: '#e67e22' }}>Adjusted</th>
                <th style={{ fontSize: '0.85rem', color: '#3498db' }}>Actual</th>
                <th style={{ fontSize: '0.85rem', color: '#e67e22' }}>Adjusted</th>
                <th style={{ fontSize: '0.85rem', color: '#3498db' }}>Actual</th>
                <th style={{ fontSize: '0.85rem', color: '#e67e22' }}>Adjusted</th>
              </tr>
            </thead>
            <tbody>
              {measurements.map((measurement, index) => {
                return (
                  <tr key={measurement.id || index}>
                    {/* Pressure - Actual */}
                    <td style={{ color: '#3498db', fontWeight: '600', fontSize: '0.9rem' }}>
                      {formatValue(measurement.pressure, 1)} 
                      <span style={{ 
                        fontSize: '0.7rem', 
                        color: 'white',
                        fontWeight: 'bold',
                        backgroundColor: getQcColor(measurement.pressure_qc),
                        padding: '0.2rem 0.4rem',
                        borderRadius: '4px',
                        marginLeft: '0.3rem'
                      }}>
                        ({measurement.pressure_qc || 'N/A'})
                      </span>
                      <div style={{ fontSize: '0.75rem', color: '#6c757d', fontWeight: 'normal' }}>
                        ~{formatValue(measurement.pressure, 0)}m depth
                      </div>
                    </td>
                    
                    {/* Pressure - Adjusted */}
                    <td style={{ color: '#e67e22', fontWeight: '600', fontSize: '0.9rem' }}>
                      {formatValue(measurement.pressure_adjusted, 1)} 
                      <span style={{ 
                        fontSize: '0.7rem', 
                        color: 'white',
                        fontWeight: 'bold',
                        backgroundColor: getQcColor(measurement.pressure_adjusted_qc),
                        padding: '0.2rem 0.4rem',
                        borderRadius: '4px',
                        marginLeft: '0.3rem'
                      }}>
                        ({measurement.pressure_adjusted_qc || 'N/A'})
                      </span>
                    </td>
                    
                    {/* Temperature - Actual */}
                    <td style={{ color: '#3498db', fontWeight: '600', fontSize: '0.9rem' }}>
                      {formatValue(measurement.temperature)}°C 
                      <span style={{ 
                        fontSize: '0.7rem', 
                        color: 'white',
                        fontWeight: 'bold',
                        backgroundColor: getQcColor(measurement.temperature_qc),
                        padding: '0.2rem 0.4rem',
                        borderRadius: '4px',
                        marginLeft: '0.3rem'
                      }}>
                        ({measurement.temperature_qc || 'N/A'})
                      </span>
                    </td>
                    
                    {/* Temperature - Adjusted */}
                    <td style={{ color: '#e67e22', fontWeight: '600', fontSize: '0.9rem' }}>
                      {formatValue(measurement.temperature_adjusted)}°C 
                      <span style={{ 
                        fontSize: '0.7rem', 
                        color: 'white',
                        fontWeight: 'bold',
                        backgroundColor: getQcColor(measurement.temperature_adjusted_qc),
                        padding: '0.2rem 0.4rem',
                        borderRadius: '4px',
                        marginLeft: '0.3rem'
                      }}>
                        ({measurement.temperature_adjusted_qc || 'N/A'})
                      </span>
                    </td>
                    
                    {/* Salinity - Actual */}
                    <td style={{ color: '#3498db', fontWeight: '600', fontSize: '0.9rem' }}>
                      {formatValue(measurement.salinity)} 
                      <span style={{ 
                        fontSize: '0.7rem', 
                        color: 'white',
                        fontWeight: 'bold',
                        backgroundColor: getQcColor(measurement.salinity_qc),
                        padding: '0.2rem 0.4rem',
                        borderRadius: '4px',
                        marginLeft: '0.3rem'
                      }}>
                        ({measurement.salinity_qc || 'N/A'})
                      </span>
                    </td>
                    
                    {/* Salinity - Adjusted */}
                    <td style={{ color: '#e67e22', fontWeight: '600', fontSize: '0.9rem' }}>
                      {formatValue(measurement.salinity_adjusted)} 
                      <span style={{ 
                        fontSize: '0.7rem', 
                        color: 'white',
                        fontWeight: 'bold',
                        backgroundColor: getQcColor(measurement.salinity_adjusted_qc),
                        padding: '0.2rem 0.4rem',
                        borderRadius: '4px',
                        marginLeft: '0.3rem'
                      }}>
                        ({measurement.salinity_adjusted_qc || 'N/A'})
                      </span>
                    </td>
                    
                    {/* Adjustment Errors */}
                    <td>
                      <div style={{ fontSize: '0.8rem', color: '#6c757d' }}>
                        {measurement.pressure_adjusted_error && (
                          <div>P: ±{formatValue(measurement.pressure_adjusted_error, 2)}</div>
                        )}
                        {measurement.temperature_adjusted_error && (
                          <div>T: ±{formatValue(measurement.temperature_adjusted_error, 3)}°C</div>
                        )}
                        {measurement.salinity_adjusted_error && (
                          <div>S: ±{formatValue(measurement.salinity_adjusted_error, 3)}</div>
                        )}
                        {!measurement.pressure_adjusted_error && !measurement.temperature_adjusted_error && !measurement.salinity_adjusted_error && (
                          <span style={{ color: '#bdc3c7' }}>N/A</span>
                        )}
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>

        {measurements.length === 0 && !loading && (
          <div className="no-data">
            <h3>📊 No Measurements Found</h3>
            <p>No measurement data is available for this profile.</p>
            <button className="profile-link-btn" onClick={onBack} style={{ marginTop: '1rem' }}>
              Select Another Profile
            </button>
          </div>
        )}

        <div className="table-info">
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginBottom: '1rem' }}>
            <div>
              <h4 style={{ color: '#2c3e50', margin: '0 0 0.5rem 0' }}>📈 Data Summary</h4>
              <p style={{ color: '#6c757d', margin: 0 }}>
                <strong>Total Measurements:</strong> {measurements.length}<br/>
                <strong>Depth Range:</strong> {measurements.length > 0 ? 
                  `${formatValue(Math.min(...measurements.filter(m => m.pressure).map(m => m.pressure)), 1)} - ${formatValue(Math.max(...measurements.filter(m => m.pressure).map(m => m.pressure)), 1)} dbar` : 
                  'N/A'}
              </p>
            </div>
            <div>
              <h4 style={{ color: '#2c3e50', margin: '0 0 0.5rem 0' }}>🔗 Data Relationship</h4>
              <p style={{ color: '#6c757d', margin: 0 }}>
                Each row represents one measurement point at a specific depth during this profile cycle. 
                The Argo float collects data while ascending or descending through the water column.
              </p>
            </div>
          </div>
          
          <div className="qc-legend">
            <h4>🏷️ Quality Control Legend:</h4>
            <div className="qc-legend-items">
              <span className="qc-legend-item">
                <span className="qc-badge" style={{ backgroundColor: '#27ae60', color: 'white' }}>1</span>
                Good quality data
              </span>
              <span className="qc-legend-item">
                <span className="qc-badge" style={{ backgroundColor: '#f39c12', color: 'white' }}>2</span>
                Probably good data
              </span>
              <span className="qc-legend-item">
                <span className="qc-badge" style={{ backgroundColor: '#e67e22', color: 'white' }}>3</span>
                Bad but correctable
              </span>
              <span className="qc-legend-item">
                <span className="qc-badge" style={{ backgroundColor: '#e74c3c', color: 'white' }}>4</span>
                Bad data
              </span>
              <span className="qc-legend-item">
                <span className="qc-badge" style={{ backgroundColor: '#8e44ad', color: 'white' }}>5</span>
                Value changed
              </span>
              <span className="qc-legend-item">
                <span className="qc-badge" style={{ backgroundColor: '#95a5a6', color: 'white' }}>8</span>
                Estimated value
              </span>
            </div>
          </div>
        </div>
      </div>

      <style jsx>{`
        .qc-badge {
          padding: 0.25rem 0.5rem;
          border-radius: 12px;
          font-size: 0.8rem;
          font-weight: 500;
          min-width: 25px;
          text-align: center;
          display: inline-block;
          cursor: help;
        }

        .no-data {
          text-align: center;
          padding: 3rem;
          color: #7f8c8d;
        }

        .table-info {
          margin-top: 1rem;
          color: #7f8c8d;
          font-size: 0.9rem;
        }

        .qc-legend {
          margin-top: 1rem;
          padding-top: 1rem;
          border-top: 1px solid #ecf0f1;
        }

        .qc-legend h4 {
          margin: 0 0 0.5rem 0;
          color: #2c3e50;
          font-size: 1rem;
        }

        .qc-legend-items {
          display: flex;
          gap: 1rem;
          flex-wrap: wrap;
        }

        .qc-legend-item {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-size: 0.85rem;
        }
      `}</style>
    </div>
  )
}

export default MeasurementsTable