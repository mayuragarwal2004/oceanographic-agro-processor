import React, { useEffect, useRef, useState, useCallback } from 'react'

// Leaflet will be loaded dynamically
let L = null

const OpenStreetMap = ({ 
  locations = [], 
  center = [0, 0], 
  zoom = 2, 
  height = '400px',
  onLocationClick = null 
}) => {
  const mapRef = useRef(null)
  const mapInstanceRef = useRef(null)
  const markersRef = useRef([])
  const gridLinesRef = useRef([])
  const oceanLabelsRef = useRef([])
  const [showGridLines, setShowGridLines] = useState(true)
  const [showOceanLabels, setShowOceanLabels] = useState(true)
  const [mapReady, setMapReady] = useState(false)
  const [boundsSet, setBoundsSet] = useState(false)
  const [currentCoords, setCurrentCoords] = useState({ lat: null, lng: null })
  const currentViewRef = useRef(null)

  // Shared function to update grid lines
  const updateGridLines = useCallback(() => {
    if (!L || !mapInstanceRef.current) return
    
    // Clear existing grid lines
    gridLinesRef.current.forEach(line => {
      if (mapInstanceRef.current.hasLayer(line)) {
        mapInstanceRef.current.removeLayer(line)
      }
    })
    gridLinesRef.current = []

    if (!showGridLines) return

    // Major latitude lines (every 30 degrees)
    const majorLatitudes = [-60, -30, 0, 30, 60]
    majorLatitudes.forEach(lat => {
      const line = L.polyline([
        [lat, -180],
        [lat, 180]
      ], {
        color: '#FF0000',
        weight: 2,
        opacity: 0.7,
        dashArray: '5, 5'
      }).addTo(mapInstanceRef.current)
      gridLinesRef.current.push(line)
      
      // Add label for major latitude lines
      const label = L.marker([lat, 0], {
        icon: L.divIcon({
          className: 'lat-lng-label',
          html: `<div style="background: rgba(255,255,255,0.8); padding: 2px 4px; border-radius: 3px; font-size: 10px; font-weight: bold; color: #FF0000;">${lat}°N</div>`,
          iconSize: [40, 20],
          iconAnchor: [20, 10]
        })
      }).addTo(mapInstanceRef.current)
      gridLinesRef.current.push(label)
    })

    // Major longitude lines (every 30 degrees)
    const majorLongitudes = [-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150]
    majorLongitudes.forEach(lng => {
      const line = L.polyline([
        [-90, lng],
        [90, lng]
      ], {
        color: '#0000FF',
        weight: 2,
        opacity: 0.7,
        dashArray: '5, 5'
      }).addTo(mapInstanceRef.current)
      gridLinesRef.current.push(line)
      
      // Add label for major longitude lines
      const label = L.marker([0, lng], {
        icon: L.divIcon({
          className: 'lat-lng-label',
          html: `<div style="background: rgba(255,255,255,0.8); padding: 2px 4px; border-radius: 3px; font-size: 10px; font-weight: bold; color: #0000FF;">${lng}°E</div>`,
          iconSize: [40, 20],
          iconAnchor: [20, 10]
        })
      }).addTo(mapInstanceRef.current)
      gridLinesRef.current.push(label)
    })

    // Add other lines (minor lines, special lines)
    // (Truncated for brevity but same logic)
    const equator = L.polyline([[0, -180], [0, 180]], {
      color: '#FF0000', weight: 3, opacity: 0.8
    }).addTo(mapInstanceRef.current)
    gridLinesRef.current.push(equator)

    const primeMeridian = L.polyline([[-90, 0], [90, 0]], {
      color: '#0000FF', weight: 3, opacity: 0.8
    }).addTo(mapInstanceRef.current)
    gridLinesRef.current.push(primeMeridian)
  }, [showGridLines])

  // Shared function to update ocean labels
  const updateOceanLabels = useCallback(() => {
    if (!L || !mapInstanceRef.current) return
    
    // Clear existing ocean labels
    oceanLabelsRef.current.forEach(label => {
      if (mapInstanceRef.current.hasLayer(label)) {
        mapInstanceRef.current.removeLayer(label)
      }
    })
    oceanLabelsRef.current = []

    if (!showOceanLabels) return

    const oceans = [
      { name: 'Pacific Ocean', lat: 0, lng: -150, style: 'font-size: 20px; font-weight: bold; color: #006994; text-shadow: 2px 2px 4px rgba(255,255,255,0.8);' },
      { name: 'Atlantic Ocean', lat: 0, lng: -30, style: 'font-size: 18px; font-weight: bold; color: #006994; text-shadow: 2px 2px 4px rgba(255,255,255,0.8);' },
      { name: 'Indian Ocean', lat: -20, lng: 80, style: 'font-size: 18px; font-weight: bold; color: #006994; text-shadow: 2px 2px 4px rgba(255,255,255,0.8);' },
      { name: 'Arctic Ocean', lat: 80, lng: 0, style: 'font-size: 16px; font-weight: bold; color: #4A90E2; text-shadow: 2px 2px 4px rgba(255,255,255,0.8);' },
      { name: 'Mediterranean Sea', lat: 35, lng: 15, style: 'font-size: 12px; font-weight: bold; color: #0080CC; text-shadow: 1px 1px 2px rgba(255,255,255,0.8);' }
    ]

    oceans.forEach(ocean => {
      const label = L.marker([ocean.lat, ocean.lng], {
        icon: L.divIcon({
          className: 'ocean-label',
          html: `<div style="background: rgba(255,255,255,0.1); padding: 4px 8px; border-radius: 4px; ${ocean.style} white-space: nowrap; pointer-events: none; transform: translate(-50%, -50%);">${ocean.name}</div>`,
          iconSize: [1, 1],
          iconAnchor: [0, 0]
        })
      }).addTo(mapInstanceRef.current)
      oceanLabelsRef.current.push(label)
    })
  }, [showOceanLabels])

  useEffect(() => {
    // Dynamically load Leaflet CSS and JS
    const loadLeaflet = async () => {
      if (!L) {
        // Load CSS
        const cssLink = document.createElement('link')
        cssLink.rel = 'stylesheet'
        cssLink.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'
        cssLink.integrity = 'sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY='
        cssLink.crossOrigin = ''
        document.head.appendChild(cssLink)

        // Load JS
        const script = document.createElement('script')
        script.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'
        script.integrity = 'sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo='
        script.crossOrigin = ''
        
        await new Promise((resolve) => {
          script.onload = resolve
          document.head.appendChild(script)
        })

        L = window.L
      }

      if (L && mapRef.current && !mapInstanceRef.current) {
        // Initialize map
        mapInstanceRef.current = L.map(mapRef.current, {
          center: center,
          zoom: zoom,
          scrollWheelZoom: true,
          dragging: true
        })

        // Add OpenStreetMap tiles
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
          attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
          maxZoom: 18
        }).addTo(mapInstanceRef.current)

        // Add mousemove event to track coordinates
        mapInstanceRef.current.on('mousemove', function(e) {
          setCurrentCoords({
            lat: e.latlng.lat.toFixed(6),
            lng: e.latlng.lng.toFixed(6)
          })
        })

        // Clear coordinates when mouse leaves the map
        mapInstanceRef.current.on('mouseout', function() {
          setCurrentCoords({ lat: null, lng: null })
        })

        // Track current view when user moves or zooms the map
        mapInstanceRef.current.on('moveend zoomend', function() {
          if (mapInstanceRef.current) {
            currentViewRef.current = {
              center: mapInstanceRef.current.getCenter(),
              zoom: mapInstanceRef.current.getZoom()
            }
          }
        })

        // Mark map as ready (grid lines and ocean labels will be initialized by their separate useEffect)
        setMapReady(true)
      }
    }

    loadLeaflet()

    return () => {
      // Cleanup map on unmount
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove()
        mapInstanceRef.current = null
      }
    }
  }, [center, zoom])

  useEffect(() => {
    if (L && mapInstanceRef.current && mapReady && locations.length > 0) {
      // Clear existing markers
      markersRef.current.forEach(marker => {
        mapInstanceRef.current.removeLayer(marker)
      })
      markersRef.current = []

      // Add new markers
      locations.forEach(location => {
        if (location.latitude && location.longitude) {
          const marker = L.marker([location.latitude, location.longitude])
            .addTo(mapInstanceRef.current)

          // Add popup with location info
          let popupContent = `
            <div style="font-size: 12px; line-height: 1.4;">
              <strong>Platform:</strong> ${location.platform_number || 'N/A'}<br>
              <strong>Cycle:</strong> ${location.cycle_number || 'N/A'}<br>
              <strong>Date:</strong> ${location.date ? new Date(location.date).toLocaleDateString() : 'N/A'}<br>
              <strong>Coordinates:</strong> ${location.latitude.toFixed(4)}°, ${location.longitude.toFixed(4)}°
          `
          
          if (location.measurement_count) {
            popupContent += `<br><strong>Measurements:</strong> ${location.measurement_count}`
          }
          
          popupContent += '</div>'
          
          marker.bindPopup(popupContent)

          // Add click handler if provided
          if (onLocationClick) {
            marker.on('click', () => onLocationClick(location))
          }

          markersRef.current.push(marker)
        }
      })

      // Fit map to show all markers only on first load
      if (markersRef.current.length > 0 && !boundsSet) {
        try {
          if (markersRef.current.length > 1) {
            const group = new L.featureGroup(markersRef.current)
            const bounds = group.getBounds()
            if (bounds && bounds.isValid() && bounds._northEast && bounds._southWest) {
              mapInstanceRef.current.fitBounds(bounds.pad(0.1))
              setBoundsSet(true)
            }
          } else if (markersRef.current.length === 1) {
            // Get the position from the first marker
            const marker = markersRef.current[0]
            const position = marker.getLatLng()
            if (position && position.lat !== undefined && position.lng !== undefined) {
              mapInstanceRef.current.setView([position.lat, position.lng], 8)
              setBoundsSet(true)
            }
          }
        } catch (error) {
          console.warn('Error setting map bounds:', error)
          // Fallback to default view if bounds setting fails
          setBoundsSet(true)
        }
      }
    }
  }, [locations, onLocationClick, mapReady, boundsSet])

  // Effect to update grid lines and ocean labels when toggle states change or map becomes ready
  useEffect(() => {
    if (mapReady) {
      updateGridLines()
      updateOceanLabels()
    }
  }, [showGridLines, showOceanLabels, updateGridLines, updateOceanLabels, mapReady])

  return (
    <div style={{ position: 'relative', width: '100%', height: height }}>
      {/* Map Legend/Control Panel */}
      <div style={{
        position: 'absolute',
        top: '10px',
        right: '10px',
        background: 'rgba(255, 255, 255, 0.95)',
        padding: '12px',
        borderRadius: '8px',
        boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)',
        zIndex: 1000,
        fontSize: '14px',
        fontFamily: 'Arial, sans-serif'
      }}>
        <div style={{ marginBottom: '8px', fontWeight: 'bold', color: '#333' }}>
          Map Options
        </div>
        
        <div style={{ marginBottom: '6px' }}>
          <label style={{ 
            display: 'flex', 
            alignItems: 'center', 
            cursor: 'pointer',
            color: '#555' 
          }}>
            <input
              type="checkbox"
              checked={showGridLines}
              onChange={(e) => setShowGridLines(e.target.checked)}
              style={{ marginRight: '8px' }}
            />
            <span style={{ display: 'flex', alignItems: 'center' }}>
              <span style={{ 
                width: '12px', 
                height: '2px', 
                background: '#FF0000', 
                marginRight: '6px',
                borderRadius: '1px'
              }}></span>
              Grid Lines
            </span>
          </label>
        </div>
        
        <div>
          <label style={{ 
            display: 'flex', 
            alignItems: 'center', 
            cursor: 'pointer',
            color: '#555' 
          }}>
            <input
              type="checkbox"
              checked={showOceanLabels}
              onChange={(e) => setShowOceanLabels(e.target.checked)}
              style={{ marginRight: '8px' }}
            />
            <span style={{ display: 'flex', alignItems: 'center' }}>
              <span style={{ 
                width: '12px', 
                height: '12px', 
                background: '#006994', 
                marginRight: '6px',
                borderRadius: '2px',
                fontSize: '8px',
                color: 'white',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}>○</span>
              Ocean Labels
            </span>
          </label>
        </div>
      </div>

      {/* Map Container */}
      <div 
        ref={mapRef} 
        style={{ 
          width: '100%', 
          height: '100%',
          borderRadius: '8px',
          border: '1px solid #ddd',
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)'
        }} 
      />

      {/* Coordinates Display */}
      {currentCoords.lat && currentCoords.lng && (
        <div style={{
          position: 'absolute',
          bottom: '10px',
          right: '10px',
          backgroundColor: 'rgba(255, 255, 255, 0.9)',
          padding: '6px 10px',
          borderRadius: '4px',
          border: '1px solid #ccc',
          fontSize: '12px',
          fontFamily: 'monospace',
          color: '#333',
          boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
          pointerEvents: 'none',
          zIndex: 1000
        }}>
          Lat: {currentCoords.lat}°, Lng: {currentCoords.lng}°
        </div>
      )}
    </div>
  )
}

export default OpenStreetMap