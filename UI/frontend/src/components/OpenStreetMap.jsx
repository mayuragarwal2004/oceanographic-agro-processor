import React, { useEffect, useRef } from 'react'

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
  }, [])

  useEffect(() => {
    if (L && mapInstanceRef.current && locations.length > 0) {
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

      // Fit map to show all markers if we have locations
      if (locations.length > 0) {
        const validLocations = locations.filter(loc => loc.latitude && loc.longitude)
        if (validLocations.length > 1) {
          const group = new L.featureGroup(markersRef.current)
          mapInstanceRef.current.fitBounds(group.getBounds().pad(0.1))
        } else if (validLocations.length === 1) {
          mapInstanceRef.current.setView(
            [validLocations[0].latitude, validLocations[0].longitude], 
            8
          )
        }
      }
    }
  }, [locations, onLocationClick])

  return (
    <div 
      ref={mapRef} 
      style={{ 
        width: '100%', 
        height: height,
        borderRadius: '8px',
        border: '1px solid #ddd',
        boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)'
      }} 
    />
  )
}

export default OpenStreetMap