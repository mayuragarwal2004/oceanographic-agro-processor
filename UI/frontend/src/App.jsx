import { useState, useEffect } from 'react'
import axios from 'axios'
import './App.css'
import ProfilesTable from './components/ProfilesTable'
import MeasurementsTable from './components/MeasurementsTable'
import Charts from './components/Charts'
import Stats from './components/Stats'

const API_BASE_URL = 'http://localhost:3001/api'

function App() {
  const [activeTab, setActiveTab] = useState('stats')
  const [selectedProfile, setSelectedProfile] = useState(null)
  const [isOnline, setIsOnline] = useState(true)

  useEffect(() => {
    // Check backend connectivity
    const checkConnectivity = async () => {
      try {
        await axios.get(`${API_BASE_URL}/stats`, { timeout: 5000 })
        setIsOnline(true)
      } catch (err) {
        setIsOnline(false)
        console.error('Backend connectivity check failed:', err)
      }
    }
    
    checkConnectivity()
    const interval = setInterval(checkConnectivity, 30000) // Check every 30 seconds
    
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="app">
      <header className="app-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1>🌊 Oceanographic Data Viewer</h1>
            <p style={{ 
              margin: '0.5rem 0 0 0', 
              color: '#6c757d', 
              fontSize: '1rem',
              fontWeight: '400'
            }}>
              Advanced Analysis of Global Argo Float Network Data
            </p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '0.5rem',
              padding: '0.5rem 1rem',
              background: isOnline ? 'rgba(39, 174, 96, 0.1)' : 'rgba(231, 76, 60, 0.1)',
              borderRadius: '20px',
              border: `1px solid ${isOnline ? '#27ae60' : '#e74c3c'}`
            }}>
              <div style={{ 
                width: '8px', 
                height: '8px', 
                borderRadius: '50%', 
                background: isOnline ? '#27ae60' : '#e74c3c' 
              }} />
              <span style={{ 
                fontSize: '0.8rem', 
                color: isOnline ? '#27ae60' : '#e74c3c',
                fontWeight: '500'
              }}>
                {isOnline ? 'Connected' : 'Offline'}
              </span>
            </div>
          </div>
        </div>
        
        <nav className="nav-tabs">
          <button 
            className={activeTab === 'stats' ? 'active' : ''} 
            onClick={() => setActiveTab('stats')}
          >
            📊 Statistics
          </button>
          <button 
            className={activeTab === 'profiles' ? 'active' : ''} 
            onClick={() => setActiveTab('profiles')}
          >
            📍 Profiles
          </button>
          <button 
            className={activeTab === 'charts' ? 'active' : ''} 
            onClick={() => setActiveTab('charts')}
          >
            📈 Charts
          </button>
          {selectedProfile && (
            <button 
              className={activeTab === 'measurements' ? 'active' : ''} 
              onClick={() => setActiveTab('measurements')}
            >
              🌊 Measurements ({selectedProfile.platform_number})
            </button>
          )}
        </nav>
      </header>

      <main className="app-main">
        {/* Breadcrumb Navigation */}
        <div style={{ 
          background: 'rgba(255, 255, 255, 0.9)', 
          padding: '1rem 1.5rem', 
          borderRadius: '12px',
          marginBottom: '1rem',
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)'
        }}>
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '0.5rem',
            fontSize: '0.9rem',
            color: '#5a6c7d'
          }}>
            <span style={{ color: '#3498db', fontWeight: '500' }}>🏠 Home</span>
            <span>→</span>
            <span style={{ 
              color: activeTab === 'stats' ? '#2c3e50' : '#3498db',
              fontWeight: activeTab === 'stats' ? '600' : '500'
            }}>
              {activeTab === 'stats' && '📊 Statistics'}
              {activeTab === 'profiles' && '📍 Profiles'}
              {activeTab === 'charts' && '📈 Charts'}
              {activeTab === 'measurements' && '🌊 Measurements'}
            </span>
            {selectedProfile && activeTab === 'measurements' && (
              <>
                <span>→</span>
                <span style={{ color: '#2c3e50', fontWeight: '600' }}>
                  Platform {selectedProfile.platform_number} - Cycle {selectedProfile.cycle_number}
                </span>
              </>
            )}
          </div>
        </div>

        {activeTab === 'stats' && <Stats />}
        {activeTab === 'profiles' && (
          <ProfilesTable 
            onProfileSelect={(profile) => {
              setSelectedProfile(profile)
              setActiveTab('measurements')
            }} 
          />
        )}
        {activeTab === 'measurements' && selectedProfile && (
          <MeasurementsTable 
            profile={selectedProfile}
            onBack={() => {
              setActiveTab('profiles')
              setSelectedProfile(null)
            }}
          />
        )}
        {activeTab === 'charts' && <Charts />}
      </main>
    </div>
  )
}

export default App
