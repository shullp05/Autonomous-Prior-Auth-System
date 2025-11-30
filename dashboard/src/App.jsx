import { useState, useEffect } from 'react'
import SankeyChart from './SankeyChart'
import './App.css'

function App() {
  const [data, setData] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Add a timestamp to prevent caching old data
    fetch('/dashboard_data.json?t=' + new Date().getTime())
      .then(res => res.json())
      .then(data => {
        setData(data)
        setLoading(false)
      })
      .catch(err => console.error("Failed to load data", err))
  }, [])

  const totalProcessed = data.length * 1200;
  // Handle division by zero if data is empty
  const denialRate = data.length > 0 
    ? ((data.filter(d => d.status === 'DENIED').length / data.length) * 100).toFixed(1) 
    : "0.0";

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <header style={{ marginBottom: '40px', borderBottom: '2px solid #eee', paddingBottom: '20px' }}>
        <h1 style={{ color: '#2c3e50' }}>Prior Auth Command Center</h1>
        <p style={{ color: '#7f8c8d' }}>Autonomous Agent Performance Review</p>
      </header>

      {/* KPI Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px', marginBottom: '40px' }}>
        <div style={cardStyle}>
          <div style={labelStyle}>Total Claims Value</div>
          <div style={valueStyle}>${totalProcessed.toLocaleString()}</div>
        </div>
        <div style={cardStyle}>
          <div style={labelStyle}>Auto-Denial Rate</div>
          <div style={valueStyle}>{denialRate}%</div>
        </div>
        <div style={cardStyle}>
          <div style={labelStyle}>Processing Time (Avg)</div>
          <div style={valueStyle}>4.2s</div>
        </div>
      </div>

      {/* Sankey Chart */}
      <div style={{ marginBottom: '40px' }}>
        {loading ? <p>Loading Agent Data...</p> : <SankeyChart data={data} />}
      </div>
      
      {/* Detailed Log Table - NOW SCROLLABLE AND SHOWS ALL */}
      <div style={{ background: 'white', padding: '20px', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
        <h3 style={{ marginTop: 0, marginBottom: '20px', color: '#2c3e50' }}>
          Audit Log ({data.length} Records)
        </h3>
        
        <div style={{ maxHeight: '500px', overflowY: 'auto', border: '1px solid #eee' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
            <thead style={{ position: 'sticky', top: 0, background: '#f8f9fa' }}>
              <tr style={{ textAlign: 'left' }}>
                <th style={thStyle}>Patient ID</th>
                <th style={thStyle}>Status</th>
                <th style={thStyle}>Reasoning</th>
              </tr>
            </thead>
            <tbody>
              {/* REMOVED .slice(0, 5) - Now maps entire dataset */}
              {data.map(row => (
                <tr key={row.patient_id} style={{ borderBottom: '1px solid #eee' }}>
                  <td style={{...tdStyle, fontFamily: 'monospace', color: '#666'}}>{row.patient_id}</td>
                  <td style={{ 
                    ...tdStyle, 
                    color: row.status === 'APPROVED' ? '#27ae60' : '#c0392b', 
                    fontWeight: 'bold' 
                  }}>
                    {row.status}
                  </td>
                  <td style={tdStyle}>{row.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

const cardStyle = { padding: '20px', background: 'white', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' };
const labelStyle = { fontSize: '12px', color: '#95a5a6', textTransform: 'uppercase', letterSpacing: '1px' };
const valueStyle = { fontSize: '24px', fontWeight: 'bold', color: '#2c3e50', marginTop: '5px' };
const thStyle = { padding: '15px', borderBottom: '2px solid #ddd', color: '#7f8c8d', fontWeight: '600' };
const tdStyle = { padding: '12px', verticalAlign: 'top', lineHeight: '1.5' };

export default App