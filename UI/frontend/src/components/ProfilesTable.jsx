import { useState, useEffect } from 'react'
import axios from 'axios'
import OpenStreetMap from './OpenStreetMap'

const API_BASE_URL = 'http://localhost:3001/api'

function ProfilesTable({ onProfileSelect }) {
  const [profiles, setProfiles] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [measurementCounts, setMeasurementCounts] = useState({})
  const [profileLocations, setProfileLocations] = useState([])
  const [showMap, setShowMap] = useState(true)

  useEffect(() => {
    fetchProfiles()
    fetchProfileLocations()
  }, [])

  const fetchProfiles = async () => {
    try {
      setLoading(true)
      const response = await axios.get(`${API_BASE_URL}/profiles?limit=100`)
      const profilesData = response.data
      setProfiles(profilesData)
      
      // Fetch measurement counts for each profile
      const counts = {}
      for (const profile of profilesData.slice(0, 10)) { // Limit to first 10 for performance
        try {
          const measurementsResponse = await axios.get(`${API_BASE_URL}/profiles/${profile.id}/measurements`)
          counts[profile.id] = measurementsResponse.data.length
        } catch (err) {
          console.error(`Error fetching measurements for profile ${profile.id}:`, err)
          counts[profile.id] = 0
        }
      }
      setMeasurementCounts(counts)
      setError(null)
    } catch (err) {
      setError('Failed to fetch profiles')
      console.error('Error fetching profiles:', err)
    } finally {
      setLoading(false)
    }
  }

  const fetchProfileLocations = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/profile-locations?limit=200`)
      setProfileLocations(response.data)
    } catch (err) {
      console.error('Error fetching profile locations:', err)
    }
  }

  const formatDate = (dateStr) => {
    return new Date(dateStr).toLocaleDateString()
  }

  const formatCoordinate = (coord) => {
    return Number(coord).toFixed(4)
  }

  if (loading) return <div className="loading">Loading profiles...</div>
  if (error) return <div className="error">{error}</div>

  return (
    <div className="profiles-container">
      <div className="card">
        <h2 style={{ color: '#2c3e50', margin: '0 0 1rem 0' }}>📍 Oceanographic Profiles Overview</h2>
        <div className="profile-summary">
          <h4>Profile-Measurement Relationship</h4>
          <p>Each profile represents one measurement cycle from an Argo float. Click any row to view its detailed temperature, salinity, and pressure measurements at different depths.</p>
        </div>
      </div>

      {/* Global Profile Locations Map */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <h3 style={{ margin: 0, color: '#2c3e50', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            🗺️ Global Profile Distribution
            <span style={{ 
              fontSize: '0.8rem', 
              background: '#e8f4fd', 
              color: '#2980b9', 
              padding: '0.25rem 0.5rem', 
              borderRadius: '12px' 
            }}>
              {profileLocations.length} locations
            </span>
          </h3>
          <button 
            onClick={() => setShowMap(!showMap)}
            style={{
              padding: '0.5rem 1rem',
              background: showMap ? '#e74c3c' : '#27ae60',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '0.9rem'
            }}
          >
            {showMap ? 'Hide Map' : 'Show Map'}
          </button>
        </div>
        
        {showMap && (
          <OpenStreetMap 
            locations={profileLocations} 
            height="300px"
            zoom={1}
            center={profileLocations.length > 0 ? [profileLocations[0].latitude, profileLocations[0].longitude] : [0, 0]}
          />
        )}
      </div>

      <div className="card">
        
        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Platform</th>
                <th>Cycle</th>
                <th>Date</th>
                <th>Location</th>
                <th>Direction</th>
                <th>Data Mode</th>
                <th>Measurements</th>
                <th>Institution</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {profiles.map((profile) => (
                <tr 
                  key={profile.id} 
                  onClick={() => onProfileSelect(profile)}
                  style={{ cursor: 'pointer' }}
                >
                  <td>
                    <strong style={{ color: '#2c3e50', fontSize: '1rem' }}>
                      {profile.platform_number}
                    </strong>
                  </td>
                  <td style={{ color: '#34495e', fontWeight: '600' }}>
                    {profile.cycle_number}
                  </td>
                  <td style={{ color: '#2c3e50' }}>
                    {formatDate(profile.profile_date)}
                  </td>
                  <td style={{ color: '#2c3e50', fontSize: '0.85rem' }}>
                    <div>
                      <strong>Lat:</strong> {formatCoordinate(profile.latitude)}°
                    </div>
                    <div>
                      <strong>Lon:</strong> {formatCoordinate(profile.longitude)}°
                    </div>
                  </td>
                  <td>
                    <span className={`status-badge ${profile.direction === 'A' ? 'ascending' : 'descending'}`}>
                      {profile.direction === 'A' ? '↑ Ascending' : profile.direction === 'D' ? '↓ Descending' : profile.direction || 'N/A'}
                    </span>
                  </td>
                  <td>
                    <span className={`status-badge ${
                      profile.data_mode === 'R' ? 'realtime' : 
                      profile.data_mode === 'D' ? 'delayed' : 
                      profile.data_mode === 'A' ? 'adjusted' : 'realtime'
                    }`}>
                      {profile.data_mode === 'R' ? 'Real-time' : 
                       profile.data_mode === 'D' ? 'Delayed' : 
                       profile.data_mode === 'A' ? 'Adjusted' : 
                       profile.data_mode || 'N/A'}
                    </span>
                  </td>
                  <td>
                    {measurementCounts[profile.id] !== undefined ? (
                      <span className="measurement-count-badge">
                        {measurementCounts[profile.id]} measurements
                      </span>
                    ) : (
                      <span style={{ color: '#6c757d', fontSize: '0.85rem' }}>Loading...</span>
                    )}
                  </td>
                  <td style={{ color: '#2c3e50', fontSize: '0.85rem' }}>
                    {profile.institution || 'N/A'}
                  </td>
                  <td>
                    <button 
                      className="profile-link-btn"
                      onClick={(e) => {
                        e.stopPropagation()
                        onProfileSelect(profile)
                      }}
                    >
                      View Data
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {profiles.length === 0 && !loading && (
          <div className="no-data">
            <h3>🌊 No Profiles Found</h3>
            <p>No oceanographic profiles are currently available in the database.</p>
          </div>
        )}

        <div className="table-info">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
            <span><strong>Showing:</strong> {profiles.length} profiles (limited to 100 for performance)</span>
            <span><strong>Tip:</strong> Click any row or "View Data" button to see detailed measurements</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ProfilesTable