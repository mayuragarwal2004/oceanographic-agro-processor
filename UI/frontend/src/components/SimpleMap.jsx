import { useState, useEffect } from 'react'

function SimpleMap({ locations, center, zoom = 2, height = 300, singleLocation = false }) {
  const [selectedLocation, setSelectedLocation] = useState(null)

  // Simple projection function for lat/lon to screen coordinates
  const projectToScreen = (lat, lon, mapWidth, mapHeight, centerLat, centerLon, zoomLevel) => {
    const scale = Math.pow(2, zoomLevel) * 100
    const x = ((lon - centerLon) * scale) + mapWidth / 2
    const y = ((centerLat - lat) * scale) + mapHeight / 2
    return { x, y }
  }

  if (!locations || locations.length === 0) {
    return (
      <div style={{ 
        height: height, 
        background: 'linear-gradient(135deg, #74b9ff 0%, #0984e3 100%)',
        borderRadius: '12px', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        color: 'white',
        fontSize: '1.1rem',
        fontWeight: '500'
      }}>
        🗺️ No location data available
      </div>
    )
  }

  const mapWidth = 600
  const mapHeight = height
  const centerLat = center ? center.lat : (singleLocation ? locations[0].latitude : 
    locations.reduce((sum, loc) => sum + parseFloat(loc.latitude), 0) / locations.length)
  const centerLon = center ? center.lon : (singleLocation ? locations[0].longitude :
    locations.reduce((sum, loc) => sum + parseFloat(loc.longitude), 0) / locations.length)

  return (
    <div style={{ 
      position: 'relative', 
      height: height,
      background: 'linear-gradient(135deg, #74b9ff 0%, #0984e3 100%)',
      borderRadius: '12px',
      overflow: 'hidden',
      border: '2px solid #ddd'
    }}>
      {/* Ocean background with subtle pattern */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: `
          radial-gradient(circle at 20% 30%, rgba(255,255,255,0.1) 2px, transparent 2px),
          radial-gradient(circle at 80% 70%, rgba(255,255,255,0.1) 1px, transparent 1px),
          radial-gradient(circle at 40% 80%, rgba(255,255,255,0.1) 1px, transparent 1px),
          linear-gradient(135deg, #74b9ff 0%, #0984e3 100%)
        `,
        backgroundSize: '50px 50px, 30px 30px, 40px 40px, 100% 100%'
      }} />

      {/* Grid lines for reference */}
      <svg style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }}>
        {/* Latitude lines */}
        {Array.from({ length: 9 }, (_, i) => (
          <line
            key={`lat-${i}`}
            x1="0"
            y1={`${(i + 1) * 10}%`}
            x2="100%"
            y2={`${(i + 1) * 10}%`}
            stroke="rgba(255,255,255,0.2)"
            strokeWidth="1"
            strokeDasharray="3,3"
          />
        ))}
        {/* Longitude lines */}
        {Array.from({ length: 9 }, (_, i) => (
          <line
            key={`lon-${i}`}
            x1={`${(i + 1) * 10}%`}
            y1="0"
            x2={`${(i + 1) * 10}%`}
            y2="100%"
            stroke="rgba(255,255,255,0.2)"
            strokeWidth="1"
            strokeDasharray="3,3"
          />
        ))}
      </svg>

      {/* Location markers */}
      {locations.map((location, index) => {
        const projected = projectToScreen(
          parseFloat(location.latitude),
          parseFloat(location.longitude),
          mapWidth,
          mapHeight,
          centerLat,
          centerLon,
          zoom
        )

        if (projected.x < 0 || projected.x > mapWidth || projected.y < 0 || projected.y > mapHeight) {
          return null // Don't render points outside the view
        }

        return (
          <div
            key={location.id || index}
            style={{
              position: 'absolute',
              left: `${(projected.x / mapWidth) * 100}%`,
              top: `${(projected.y / mapHeight) * 100}%`,
              transform: 'translate(-50%, -50%)',
              cursor: 'pointer',
              zIndex: selectedLocation === index ? 1000 : 100
            }}
            onClick={() => setSelectedLocation(selectedLocation === index ? null : index)}
          >
            {/* Marker */}
            <div style={{
              width: singleLocation ? '20px' : '12px',
              height: singleLocation ? '20px' : '12px',
              background: singleLocation ? '#e74c3c' : '#f39c12',
              border: '2px solid white',
              borderRadius: '50%',
              boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
              transition: 'all 0.3s ease',
              transform: selectedLocation === index ? 'scale(1.5)' : 'scale(1)'
            }} />

            {/* Tooltip */}
            {selectedLocation === index && (
              <div style={{
                position: 'absolute',
                bottom: '30px',
                left: '50%',
                transform: 'translateX(-50%)',
                background: 'rgba(0,0,0,0.9)',
                color: 'white',
                padding: '8px 12px',
                borderRadius: '8px',
                fontSize: '0.8rem',
                whiteSpace: 'nowrap',
                zIndex: 1001,
                boxShadow: '0 4px 12px rgba(0,0,0,0.3)'
              }}>
                <div><strong>{location.platform_number || 'Platform'}</strong></div>
                <div>{parseFloat(location.latitude).toFixed(3)}°N, {parseFloat(location.longitude).toFixed(3)}°E</div>
                {location.profile_date && (
                  <div>{new Date(location.profile_date).toLocaleDateString()}</div>
                )}
                {location.institution && (
                  <div style={{ fontSize: '0.7rem', opacity: 0.8 }}>{location.institution}</div>
                )}
              </div>
            )}
          </div>
        )
      })}

      {/* Legend */}
      <div style={{
        position: 'absolute',
        bottom: '10px',
        left: '10px',
        background: 'rgba(255,255,255,0.95)',
        padding: '8px 12px',
        borderRadius: '8px',
        fontSize: '0.8rem',
        color: '#2c3e50',
        boxShadow: '0 2px 8px rgba(0,0,0,0.2)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
          <div style={{ 
            width: '12px', 
            height: '12px', 
            background: singleLocation ? '#e74c3c' : '#f39c12', 
            borderRadius: '50%',
            border: '1px solid white'
          }} />
          <span>{singleLocation ? 'Current Profile' : `${locations.length} Profiles`}</span>
        </div>
        <div style={{ fontSize: '0.7rem', opacity: 0.7 }}>Click markers for details</div>
      </div>

      {/* Compass */}
      <div style={{
        position: 'absolute',
        top: '10px',
        right: '10px',
        background: 'rgba(255,255,255,0.95)',
        padding: '8px',
        borderRadius: '50%',
        fontSize: '1.2rem',
        boxShadow: '0 2px 8px rgba(0,0,0,0.2)'
      }}>
        🧭
      </div>
    </div>
  )
}

export default SimpleMap