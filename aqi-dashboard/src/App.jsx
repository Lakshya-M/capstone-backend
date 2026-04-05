import { useState, useRef, useEffect } from 'react'
import {
  getInitialBuildingState,
  ROOM_IDS_LIST,
  getVariableValuesForRoom,
  getNodesForRoom,
  getRoomStatus,
  getAverageVoltageForRoom,
  updateNodeInBuildingState,
} from './buildingState'
import {
  runSimulationCycle,
  nudgeRoomTrends,
  checkAnomaly,
  SIMULATION_INTERVAL_MS,
  TREND_NUDGE_INTERVAL_MS,
  PLAYBACK_SNAPSHOT_MAX,
} from './simulationEngine'

// ViewBox and geometry (from your SVG)
const VIEWBOX = { w: 1920, h: 1080 }
const ROOM = { x: 247.5, y: 157.94, width: 1428, height: 762 }

const SENSORS = [
  { x: 275.3, y: 184 },
  { x: 275.3, y: 900 },
  { x: 1399.22, y: 625.63 },
]

const CANVAS_W = 480
const CANVAS_H = 270
const SIGMA = 280

// Single color scale: 0–300 mapped to Blue → Green → Yellow → Orange → Red
const HEAT_COLORS = [
  { t: 0, r: 0, g: 100, b: 255 },
  { t: 50, r: 0, g: 228, b: 0 },
  { t: 100, r: 255, g: 255, b: 0 },
  { t: 150, r: 255, g: 126, b: 0 },
  { t: 200, r: 255, g: 0, b: 0 },
  { t: 300, r: 180, g: 0, b: 50 },
]

function valueToColor(normalized) {
  const v = Math.max(0, Math.min(300, Number(normalized) || 0))
  for (let i = 0; i < HEAT_COLORS.length - 1; i++) {
    const a = HEAT_COLORS[i]
    const b = HEAT_COLORS[i + 1]
    if (v <= b.t) {
      const t = (v - a.t) / (b.t - a.t)
      const r = Math.round(a.r + (b.r - a.r) * t)
      const g = Math.round(a.g + (b.g - a.g) * t)
      const bl = Math.round(a.b + (b.b - a.b) * t)
      return `rgb(${r},${g},${bl})`
    }
  }
  const last = HEAT_COLORS[HEAT_COLORS.length - 1]
  return `rgb(${last.r},${last.g},${last.b})`
}

function gaussianWeight(d, sigma) {
  return Math.exp(-(d * d) / (2 * sigma * sigma))
}

function blendedValue(vx, vy, sensorValues) {
  let sumW = 0
  let sumWV = 0
  for (let i = 0; i < SENSORS.length; i++) {
    const dx = vx - SENSORS[i].x
    const dy = vy - SENSORS[i].y
    const d = Math.sqrt(dx * dx + dy * dy)
    const w = gaussianWeight(d, SIGMA)
    const val = Number(sensorValues[i]) || 0
    sumW += w
    sumWV += w * val
  }
  if (sumW < 1e-9) return 0
  return sumWV / sumW
}

// Normalize to 0–300 for color scale: AQI 0–300, Temperature 0–50°C, Humidity 0–100%
function normalizeForColor(value, variable) {
  const v = Number(value) || 0
  if (variable === 'aqi') return Math.min(300, v)
  if (variable === 'temperature') return (v / 50) * 300
  if (variable === 'humidity') return (v / 100) * 300
  return v
}

/** Map normalized scale 0–300 back to raw units for legend text */
function formatLegendRange(nMin, nMax, variable) {
  if (variable === 'aqi') {
    return `${Math.round(nMin)}–${Math.round(nMax)}`
  }
  if (variable === 'temperature') {
    const tMin = (nMin * 50) / 300
    const tMax = (nMax * 50) / 300
    return `${tMin.toFixed(1)}–${tMax.toFixed(1)} °C`
  }
  if (variable === 'humidity') {
    const hMin = (nMin * 100) / 300
    const hMax = (nMax * 100) / 300
    return `${hMin.toFixed(1)}–${hMax.toFixed(1)} %`
  }
  return `${Math.round(nMin)}–${Math.round(nMax)}`
}

const HEAT_LEGEND_SEGMENTS = [
  { label: 'Low', nMin: 0, nMax: 50, color: 'rgb(0,100,255)' },
  { label: 'Moderate', nMin: 50, nMax: 100, color: '#00e400' },
  { label: 'Elevated', nMin: 100, nMax: 150, color: '#ffff00' },
  { label: 'High', nMin: 150, nMax: 200, color: '#ff7e00' },
  { label: 'Very high', nMin: 200, nMax: 300, color: '#ff0000' },
]

function drawHeatmap(ctx, sensorValues, variable) {
  ctx.clearRect(0, 0, CANVAS_W, CANVAS_H)
  const scaleX = VIEWBOX.w / CANVAS_W
  const scaleY = VIEWBOX.h / CANVAS_H
  const roomRight = ROOM.x + ROOM.width
  const roomBottom = ROOM.y + ROOM.height

  for (let py = 0; py < CANVAS_H; py++) {
    for (let px = 0; px < CANVAS_W; px++) {
      const vx = (px + 0.5) * scaleX
      const vy = (py + 0.5) * scaleY
      if (vx < ROOM.x || vx > roomRight || vy < ROOM.y || vy > roomBottom) continue
      const raw = blendedValue(vx, vy, sensorValues)
      const normalized = normalizeForColor(raw, variable)
      ctx.fillStyle = valueToColor(normalized)
      ctx.globalAlpha = 0.82
      ctx.fillRect(px, py, 1, 1)
    }
  }
  ctx.globalAlpha = 1
}

const HEATMAP_VARIABLES = [
  { id: 'aqi', label: 'AQI' },
  { id: 'temperature', label: 'Temperature' },
  { id: 'humidity', label: 'Humidity' },
]

const ROOM_TITLES = { roomA: 'Room 1', roomB: 'Room 2', roomC: 'Room 3', roomD: 'Room 4' }

const HEATMAP_TRANSITION_MS = 500

function easeOutCubic(t) {
  return 1 - (1 - t) ** 3
}

function interpolateValues(prev, target, progress) {
  return prev.map((p, i) => (Number(p) || 0) + ((Number(target[i]) || 0) - (Number(p) || 0)) * progress)
}

const TABS = [
  { id: 'simulation', label: 'Simulation' },
  { id: 'anomaly', label: 'Anomaly Detection' },
  { id: 'playback', label: 'Playback' },
  { id: 'alerts', label: 'Alerts' },
  { id: 'fault', label: 'Fault Injection' },
  { id: 'voltage', label: 'Voltage' },
]

const VOLTAGE_HISTORY_LEN = 30
const VOLTAGE_NORMAL_MIN = 200
const VOLTAGE_NORMAL_MAX = 250

/** Max alert rows kept in memory / shown in log */
const ALERTS_STORED_MAX = 100

/** Tab badge: show exact count below 100, then 100+, 150+, 200+, … (cumulative session total). */
function formatAlertBadgeCount(total) {
  if (total < 100) return String(total)
  const base = 100 + Math.floor((total - 100) / 50) * 50
  return `${base}+`
}

/** FastAPI twin store for hardware node D1 (Room 4); same host as energy API */
const TWIN_LATEST_URL = 'http://127.0.0.1:8000/api/twin/latest'

const FAULT_TYPES = [
  { id: 'high_aqi', label: 'High AQI', values: { aqi: 250 } },
  { id: 'high_temperature', label: 'High Temperature', values: { temperature: 38 } },
  { id: 'high_humidity', label: 'High Humidity', values: { humidity: 90 } },
  { id: 'voltage_spike', label: 'Voltage Spike', values: { voltage: 270 } },
]

export default function App() {
  const [buildingState, setBuildingState] = useState(getInitialBuildingState)
  const [heatmapVariable, setHeatmapVariable] = useState('aqi')
  const [alerts, setAlerts] = useState([])
  /** Cumulative alert events this session (badge); log list is capped at ALERTS_STORED_MAX */
  const [alertTotalCount, setAlertTotalCount] = useState(0)
  const [activeTab, setActiveTab] = useState('simulation')
  const [playbackLength, setPlaybackLength] = useState(0)
  const [playbackIndex, setPlaybackIndex] = useState(0)
  const [faultRoom, setFaultRoom] = useState('roomA')
  const [faultNode, setFaultNode] = useState('A1')
  const [faultType, setFaultType] = useState('high_aqi')
  const [voltageHistory, setVoltageHistory] = useState(() =>
    Object.fromEntries(ROOM_IDS_LIST.map((id) => [id, []]))
  )
  const [energyData, setEnergyData] = useState(null)
  const [energyError, setEnergyError] = useState(null)
  const [mlTestResult, setMlTestResult] = useState(null)
  const playbackSnapshotsRef = useRef([])
  const prevEnergyAnomalyRef = useRef(false)
  const energyFetchInFlightRef = useRef(false)

  // Sensor simulation: every 2s update non-hardware nodes, merge ESP32 D1 if API has data, then alerts + snapshot
  useEffect(() => {
    const tick = async () => {
      let twinPatch = null
      try {
        const res = await fetch(TWIN_LATEST_URL)
        if (res.ok) {
          const j = await res.json()
          if (j.temperature != null && j.humidity != null && j.aqi != null) {
            twinPatch = {
              aqi: Math.round(Number(j.aqi)),
              temperature: Number(Number(j.temperature).toFixed(1)),
              humidity: Math.round(Number(j.humidity)),
            }
            if (j.voltage != null) twinPatch.voltage = Number(j.voltage)
          }
        }
      } catch {
        /* Backend offline: D1 keeps previous values */
      }

      setBuildingState((prev) => {
        let base = prev
        if (twinPatch) base = updateNodeInBuildingState(prev, 'roomD', 'D1', twinPatch)
        const { nextState, alerts: newAlerts } = runSimulationCycle(base)
        setAlerts((a) => [...newAlerts, ...a].slice(0, ALERTS_STORED_MAX))
        if (newAlerts.length > 0) {
          setAlertTotalCount((t) => t + newAlerts.length)
        }
        setVoltageHistory((vh) => {
          const next = { ...vh }
          for (const roomId of ROOM_IDS_LIST) {
            const avg = getAverageVoltageForRoom(nextState, roomId)
            next[roomId] = [...(next[roomId] || []).slice(-(VOLTAGE_HISTORY_LEN - 1)), avg]
          }
          return next
        })
        playbackSnapshotsRef.current = [
          ...playbackSnapshotsRef.current.slice(-(PLAYBACK_SNAPSHOT_MAX - 1)),
          { state: nextState, alerts: newAlerts, timestamp: Date.now() },
        ]
        setPlaybackLength(playbackSnapshotsRef.current.length)
        return nextState
      })
    }
    const intervalId = setInterval(tick, SIMULATION_INTERVAL_MS)
    return () => clearInterval(intervalId)
  }, [])

  // Environmental drift: nudge per-room trend values every 30s so baseline drift evolves
  useEffect(() => {
    const intervalId = setInterval(nudgeRoomTrends, TREND_NUDGE_INTERVAL_MS)
    return () => clearInterval(intervalId)
  }, [])

  async function fetchEnergyData() {
    if (energyFetchInFlightRef.current) return
    energyFetchInFlightRef.current = true
    try {
      const res = await fetch('http://127.0.0.1:8000/api/energy-data')
      if (!res.ok) throw new Error('Server returned ' + res.status)
      const data = await res.json()

      setEnergyError(null)
      setEnergyData(data)

      var isAnomalous = data.severity === 'moderate' || data.severity === 'high'
      if (isAnomalous && !prevEnergyAnomalyRef.current) {
        prevEnergyAnomalyRef.current = true
        setAlerts(function (a) {
          return [
            {
              type: 'ml_energy_anomaly',
              roomId: null,
              nodeId: null,
              timestamp: Date.now(),
              power: data.power,
              energy: data.energy,
              deviation_percent: data.deviation_percent,
              severity: data.severity,
            },
            ...a,
          ].slice(0, ALERTS_STORED_MAX)
        })
        setAlertTotalCount((t) => t + 1)
      }
      if (!isAnomalous) prevEnergyAnomalyRef.current = false
    } catch (err) {
      setEnergyError(err.message || 'Cannot reach energy API')
    } finally {
      energyFetchInFlightRef.current = false
    }
  }

  useEffect(() => {
    fetchEnergyData()
    const intervalId = setInterval(fetchEnergyData, 2000)
    return () => clearInterval(intervalId)
  }, [])

  const handlePlaybackSliderChange = (e) => {
    const index = parseInt(e.target.value, 10)
    setPlaybackIndex(index)
    const snapshot = playbackSnapshotsRef.current[index]
    if (snapshot?.state) setBuildingState(snapshot.state)
  }

  const faultTypeConfig = FAULT_TYPES.find((f) => f.id === faultType)
  const faultNodes = getNodesForRoom(buildingState, faultRoom)
  const validFaultNode = faultNodes.some((n) => n.id === faultNode) ? faultNode : faultNodes[0]?.id ?? faultNode

  const handleFaultInject = () => {
    if (!faultTypeConfig) return
    const nodeId = faultNodes.some((n) => n.id === faultNode) ? faultNode : faultNodes[0]?.id
    if (!nodeId) return
    setBuildingState((prev) => {
      const next = updateNodeInBuildingState(prev, faultRoom, nodeId, faultTypeConfig.values)
      const node = next.building?.rooms?.[faultRoom]?.nodes?.[nodeId]
      if (!node) return prev
      const anomaly = checkAnomaly(node)
      if (anomaly) {
        node.status = 'alert'
        const room = next.building?.rooms?.[faultRoom]
        if (room) room.status = 'alert'
        setAlerts((a) => [{ roomId: faultRoom, nodeId, ...anomaly, timestamp: Date.now() }, ...a].slice(0, ALERTS_STORED_MAX))
        setAlertTotalCount((t) => t + 1)
      }

      // If user injects a voltage fault, immediately push the room's average voltage
      // into voltageHistory so the voltage graphs reflect the change right away.
      if (faultTypeConfig.values && faultTypeConfig.values.voltage != null) {
        const room = next.building?.rooms?.[faultRoom]
        if (room?.nodes) {
          const nodeValues = Object.values(room.nodes).map((n) => Number(n.voltage) || 0)
          const avg = Number((nodeValues.reduce((s, v) => s + v, 0) / Math.max(1, nodeValues.length)).toFixed(1))
          setVoltageHistory((vh) => {
            const prevHistory = vh[faultRoom] || []
            const nextHistory = [...prevHistory.slice(-(VOLTAGE_HISTORY_LEN - 1)), avg]
            return { ...vh, [faultRoom]: nextHistory }
          })
        }
      }
      return next
    })
  }

  return (
    <div className="dashboard">
      <h1>Smart Building — Digital Twin</h1>

      {/* Heatmap variable selector — always visible */}
      <div className="dashboard-controls">
        <label className="variable-selector-label">
          Heatmap Variable
          <select
            value={heatmapVariable}
            onChange={(e) => setHeatmapVariable(e.target.value)}
            className="variable-selector"
          >
            {HEATMAP_VARIABLES.map((v) => (
              <option key={v.id} value={v.id}>
                {v.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      {/* Tab bar */}
      <div className="tab-bar" role="tablist">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            role="tab"
            aria-selected={activeTab === tab.id}
            aria-controls={`panel-${tab.id}`}
            id={`tab-${tab.id}`}
            className={`tab-button ${activeTab === tab.id ? 'tab-button--active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
            {tab.id === 'alerts' && alertTotalCount > 0 && (
              <span className="tab-badge" title={`${alertTotalCount} alert events this session`}>
                {formatAlertBadgeCount(alertTotalCount)}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Tab panels — only active panel visible */}
      <div className="tab-panels">
        {activeTab === 'simulation' && (
          <div id="panel-simulation" role="tabpanel" aria-labelledby="tab-simulation" className="tab-panel">
            <SimulationPanel intervalMs={SIMULATION_INTERVAL_MS} />
          </div>
        )}
        {activeTab === 'anomaly' && (
          <div id="panel-anomaly" role="tabpanel" aria-labelledby="tab-anomaly" className="tab-panel">
            <AnomalyPanel buildingState={buildingState} />
          </div>
        )}
        {activeTab === 'playback' && (
          <div id="panel-playback" role="tabpanel" aria-labelledby="tab-playback" className="tab-panel">
            <PlaybackPanel
              playbackLength={playbackLength}
              playbackIndex={playbackIndex}
              onSliderChange={handlePlaybackSliderChange}
            />
          </div>
        )}
        {activeTab === 'alerts' && (
          <div id="panel-alerts" role="tabpanel" aria-labelledby="tab-alerts" className="tab-panel">
            <AlertsPanel alerts={alerts} />
          </div>
        )}
        {activeTab === 'voltage' && (
          <div id="panel-voltage" role="tabpanel" aria-labelledby="tab-voltage" className="tab-panel">
            <VoltagePanel
              buildingState={buildingState}
              voltageHistory={voltageHistory}
              roomIds={ROOM_IDS_LIST}
              roomTitles={ROOM_TITLES}
              normalMin={VOLTAGE_NORMAL_MIN}
              normalMax={VOLTAGE_NORMAL_MAX}
            />
          </div>
        )}
        {activeTab === 'fault' && (
          <div id="panel-fault" role="tabpanel" aria-labelledby="tab-fault" className="tab-panel">
            <FaultInjectionPanel
              roomIds={ROOM_IDS_LIST}
              roomTitles={ROOM_TITLES}
              faultRoom={faultRoom}
              faultNode={validFaultNode}
              faultType={faultType}
              faultTypes={FAULT_TYPES}
              faultNodes={faultNodes}
              onRoomChange={(id) => {
                setFaultRoom(id)
                const nodes = getNodesForRoom(buildingState, id)
                setFaultNode(nodes[0]?.id ?? '')
              }}
              onNodeChange={setFaultNode}
              onFaultTypeChange={setFaultType}
              onInject={handleFaultInject}
            />
          </div>
        )}
      </div>

      {/* 2x2 grid of room panels — visualization only, no node values */}
      <div className="rooms-grid">
        {ROOM_IDS_LIST.map((roomId) => (
          <RoomPanel
            key={roomId}
            roomId={roomId}
            title={ROOM_TITLES[roomId]}
            buildingState={buildingState}
            variable={heatmapVariable}
          />
        ))}
      </div>

      {/* Energy Monitoring & ML Anomaly Detection — below room grid */}
      <EnergyMlPanel
        energyData={energyData}
        energyError={energyError}
        mlTestResult={mlTestResult}
        setMlTestResult={setMlTestResult}
        onRunMlTest={async (testPower, testEnergy) => {
          try {
            const res = await fetch('http://127.0.0.1:8000/api/test-anomaly', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ power: testPower, energy: testEnergy }),
            })
            if (!res.ok) {
              const txt = await res.text()
              setMlTestResult({ error: 'Server error: ' + (txt || res.status) })
              return
            }
            const data = await res.json()
            setMlTestResult(data)
            if (data.severity === 'moderate' || data.severity === 'high') {
              setAlerts((a) => [
                {
                  type: 'ml_energy_anomaly',
                  roomId: null,
                  nodeId: null,
                  timestamp: Date.now(),
                  power: data.power,
                  energy: data.energy,
                  deviation_percent: data.deviation_percent,
                  severity: data.severity,
                },
                ...a,
              ].slice(0, ALERTS_STORED_MAX))
              setAlertTotalCount((t) => t + 1)
            }
          } catch (err) {
            setMlTestResult({ error: err.message || 'Cannot reach energy API' })
          }
        }}
      />

      <div className="legend" aria-label="Heatmap color scale">
        <p className="legend-heading">
          Scale for <strong>{HEATMAP_VARIABLES.find((v) => v.id === heatmapVariable)?.label ?? heatmapVariable}</strong>
        </p>
        <div className="legend-row">
          {HEAT_LEGEND_SEGMENTS.map((seg) => (
            <span key={seg.label} className="legend-item">
              <span className="legend-box" style={{ background: seg.color }} />
              <span className="legend-item-text">
                <span className="legend-label">{seg.label}</span>
                <span className="legend-range">{formatLegendRange(seg.nMin, seg.nMax, heatmapVariable)}</span>
              </span>
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}

function SimulationPanel({ intervalMs }) {
  return (
    <div className="tab-panel-content">
      <h3 className="tab-panel-title">Simulation controls</h3>
      <p className="tab-panel-text">
        Sensor simulation runs automatically every <strong>{intervalMs / 1000}s</strong>. Non-hardware nodes (all except D1) are updated with gradual deltas plus per-room baseline drift (AQI ±1, temp ±0.1°C, humidity ±0.3% per cycle). Drift trends are nudged every 30s for natural evolution. Heatmaps re-render each cycle.
      </p>
      <ul className="tab-panel-list">
        <li>AQI: ±8 per cycle (clamp 20–200)</li>
        <li>Temperature: ±0.8°C per cycle (clamp 18–35)</li>
        <li>Humidity: ±2% per cycle (clamp 30–85)</li>
        <li>Voltage: ±2 V per cycle (clamp 210–240)</li>
      </ul>
    </div>
  )
}

function AnomalyPanel({ buildingState }) {
  const roomStatus = ROOM_IDS_LIST.map((roomId) => {
    const nodes = getNodesForRoom(buildingState, roomId)
    const alertCount = nodes.filter((n) => n.status === 'alert').length
    const nodeSummary = nodes.map((n) => ({ id: n.id, status: n.status }))
    return { roomId, title: ROOM_TITLES[roomId], alertCount, nodes: nodeSummary }
  })
  return (
    <div className="tab-panel-content">
      <h3 className="tab-panel-title">Room status</h3>
      <div className="room-status-grid">
        {roomStatus.map((room) => (
          <div key={room.roomId} className="room-status-card">
            <h4 className="room-status-title">{room.title}</h4>
            <p className={`room-status-badge ${room.alertCount > 0 ? 'room-status-badge--alert' : 'room-status-badge--normal'}`}>
              {room.alertCount > 0 ? `${room.alertCount} alert(s)` : 'Normal'}
            </p>
            <ul className="room-status-nodes">
              {room.nodes.map((n) => (
                <li key={n.id} className={n.status === 'alert' ? 'node-alert' : ''}>
                  {n.id}: {n.status}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  )
}

function PlaybackPanel({ playbackLength, playbackIndex, onSliderChange }) {
  const max = Math.max(0, playbackLength - 1)
  return (
    <div className="tab-panel-content">
      <h3 className="tab-panel-title">Playback</h3>
      <p className="tab-panel-text">
        Scrub through recorded snapshots. Moving the slider updates the dashboard to that moment.
      </p>
      <div className="playback-controls">
        <label className="playback-label">
          Frame {playbackIndex + 1} of {playbackLength || 1}
        </label>
        <input
          type="range"
          min={0}
          max={max}
          value={Math.min(playbackIndex, max)}
          onChange={onSliderChange}
          className="playback-slider"
          disabled={playbackLength === 0}
        />
      </div>
    </div>
  )
}

const ALERT_LOG_MAX = ALERTS_STORED_MAX

const ALERT_TYPE_LABELS = {
  air_quality_alert: 'High AQI',
  temperature_alert: 'High Temperature',
  humidity_alert: 'High Humidity',
  power_alert: 'Voltage Spike',
  ml_energy_anomaly: 'ML Energy Anomaly',
}

function getAlertSeverity(type, mlSeverity) {
  if (type === 'ml_energy_anomaly') return mlSeverity === 'high' ? 'critical' : 'warning'
  if (type === 'humidity_alert') return 'warning'
  return 'critical'
}

function formatAlertLogEntry(a) {
  const time = new Date(a.timestamp).toLocaleTimeString()
  const label = ALERT_TYPE_LABELS[a.type] || a.type?.replace(/_/g, ' ') || 'Alert'
  if (a.type === 'ml_energy_anomaly') {
    const lines = [`[${time}] Energy anomaly detected`]
    if (a.power != null) lines.push(`Power: ${Number(a.power).toFixed(1)} W`)
    if (a.energy != null) lines.push(`Energy: ${Number(a.energy).toFixed(3)} kWh`)
    if (a.deviation_percent != null) lines.push(`Deviation: ${Number(a.deviation_percent).toFixed(1)}%`)
    return lines.join('\n')
  }
  const room = ROOM_TITLES[a.roomId] || a.roomId
  return `[${time}] ${room} Node ${a.nodeId} ${label}`
}

function AlertsPanel({ alerts }) {
  const entries = alerts.slice(0, ALERT_LOG_MAX)
  return (
    <div className="tab-panel-content">
      <h3 className="tab-panel-title">Alert log</h3>
      <p className="tab-panel-text">
        Latest {ALERT_LOG_MAX} entries (newest at top). Tab badge is total events this session; hover badge for exact count.
      </p>
      {entries.length === 0 ? (
        <p className="tab-panel-text tab-panel-text--muted">No alert entries yet.</p>
      ) : (
        <ul className="alert-log">
          {entries.map((a, i) => {
            const severity = getAlertSeverity(a.type, a.severity)
            return (
              <li
                key={`${a.roomId}-${a.nodeId}-${a.timestamp}-${i}`}
                className={`alert-log-item alert-log-item--${severity}`}
                style={{ whiteSpace: 'pre-line' }}
              >
                {formatAlertLogEntry(a)}
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}

function FaultInjectionPanel({
  roomIds,
  roomTitles,
  faultRoom,
  faultNode,
  faultType,
  faultTypes,
  faultNodes,
  onRoomChange,
  onNodeChange,
  onFaultTypeChange,
  onInject,
}) {
  return (
    <div className="tab-panel-content">
      <h3 className="tab-panel-title">Fault injection</h3>
      <p className="tab-panel-text">
        Inject a fault into a node to trigger anomaly detection and alert. Heatmap updates immediately.
      </p>
      <div className="fault-controls">
        <label className="fault-control">
          Select Room
          <select value={faultRoom} onChange={(e) => onRoomChange(e.target.value)} className="fault-select">
            {roomIds.map((id) => (
              <option key={id} value={id}>{roomTitles[id] ?? id}</option>
            ))}
          </select>
        </label>
        <label className="fault-control">
          Select Node
          <select
            value={faultNode}
            onChange={(e) => onNodeChange(e.target.value)}
            className="fault-select"
          >
            {faultNodes.map((n) => (
              <option key={n.id} value={n.id}>{n.id}</option>
            ))}
          </select>
        </label>
        <label className="fault-control">
          Select Fault Type
          <select value={faultType} onChange={(e) => onFaultTypeChange(e.target.value)} className="fault-select">
            {faultTypes.map((f) => (
              <option key={f.id} value={f.id}>{f.label}</option>
            ))}
          </select>
        </label>
        <button type="button" className="btn btn--inject" onClick={onInject}>
          Inject fault
        </button>
      </div>
      <ul className="tab-panel-list">
        <li>High AQI → AQI = 250</li>
        <li>High Temperature → temp = 38°C</li>
        <li>High Humidity → humidity = 90%</li>
        <li>Voltage Spike → voltage = 270V</li>
      </ul>
    </div>
  )
}

function severityBadgeClass(severity) {
  if (severity === 'high') return 'energy-ml-status-badge--high'
  if (severity === 'moderate') return 'energy-ml-status-badge--moderate'
  return 'energy-ml-status-badge--normal'
}

function severityLabel(status, severity) {
  if (severity === 'high') return 'HIGH ALERT'
  if (severity === 'moderate') return 'MODERATE'
  return status || 'NORMAL'
}

function EnergyMlPanel({ energyData, energyError, mlTestResult, setMlTestResult, onRunMlTest }) {
  const [testPower, setTestPower] = useState('')
  const [testEnergy, setTestEnergy] = useState('')
  const isAnomalous = energyData?.severity === 'moderate' || energyData?.severity === 'high'

  const handleRunTest = () => {
    const power = parseFloat(testPower)
    const energy = parseFloat(testEnergy)
    if (Number.isFinite(power) && Number.isFinite(energy)) {
      onRunMlTest(power, energy)
    } else if (setMlTestResult) {
      setMlTestResult({ error: 'Enter valid Power (W) and Energy (kWh)' })
    }
  }

  return (
    <div className={`energy-ml-panel ${isAnomalous ? 'energy-ml-panel--anomaly' : ''}`}>
      <h2 className="energy-ml-panel-title">
        Energy Monitoring &amp; ML Anomaly Detection
        {isAnomalous && <span className="energy-ml-warning-badge">{energyData?.severity === 'high' ? 'HIGH ALERT' : 'MODERATE'}</span>}
      </h2>
      <div className="energy-ml-content">
        {/* Live energy metrics */}
        <div className="energy-ml-metrics">
          <h3 className="energy-ml-subtitle">Live energy metrics</h3>
          {energyError ? (
            <p className="energy-ml-error">{energyError}</p>
          ) : (
            <div className="energy-ml-values">
              <div className="energy-ml-row">
                <span className="energy-ml-label">Voltage (V)</span>
                <span className="energy-ml-value">{energyData != null && energyData.voltage != null ? Number(energyData.voltage).toFixed(1) : '—'}</span>
              </div>
              <div className="energy-ml-row">
                <span className="energy-ml-label">Power (W)</span>
                <span className="energy-ml-value">{energyData != null ? Number(energyData.power).toFixed(1) : '—'}</span>
              </div>
              <div className="energy-ml-row">
                <span className="energy-ml-label">Energy (kWh)</span>
                <span className="energy-ml-value">{energyData != null ? Number(energyData.energy).toFixed(4) : '—'}</span>
              </div>
              <div className="energy-ml-row">
                <span className="energy-ml-label">Recon. Error</span>
                <span className="energy-ml-value">{energyData != null ? Number(energyData.reconstruction_error).toFixed(4) : '—'}</span>
              </div>
              <div className="energy-ml-row">
                <span className="energy-ml-label">Threshold</span>
                <span className="energy-ml-value">{energyData != null ? Number(energyData.threshold).toFixed(4) : '—'}</span>
              </div>
              <div className="energy-ml-row">
                <span className="energy-ml-label">Deviation %</span>
                <span className="energy-ml-value">{energyData != null ? Number(energyData.deviation_percent).toFixed(1) + '%' : '—'}</span>
              </div>
            </div>
          )}
        </div>
        {/* ML status badge */}
        <div className="energy-ml-status">
          <h3 className="energy-ml-subtitle">ML anomaly status</h3>
          <p className={`energy-ml-status-badge ${energyData != null ? severityBadgeClass(energyData.severity) : 'energy-ml-status-badge--normal'}`}>
            {energyData == null && !energyError ? '—' : severityLabel(energyData?.status, energyData?.severity)}
          </p>
        </div>
        {/* Manual test injection */}
        <div className="energy-ml-test">
          <h3 className="energy-ml-subtitle">Manual test injection</h3>
          <div className="energy-ml-test-inputs">
            <label className="energy-ml-test-label">
              Power (W)
              <input
                type="number"
                step="any"
                value={testPower}
                onChange={(e) => setTestPower(e.target.value)}
                className="energy-ml-input"
                placeholder="e.g. 24.2"
              />
            </label>
            <label className="energy-ml-test-label">
              Energy (kWh)
              <input
                type="number"
                step="any"
                value={testEnergy}
                onChange={(e) => setTestEnergy(e.target.value)}
                className="energy-ml-input"
                placeholder="e.g. 0.031"
              />
            </label>
            <button type="button" className="btn btn--ml-test" onClick={handleRunTest}>
              Run ML Test
            </button>
          </div>
          {mlTestResult && (
            <div className="energy-ml-test-result">
              {mlTestResult.error ? (
                <p className="energy-ml-error">{mlTestResult.error}</p>
              ) : (
                <div className={`energy-ml-test-card ${mlTestResult.severity === 'high' ? 'energy-ml-test-card--high' : mlTestResult.severity === 'moderate' ? 'energy-ml-test-card--moderate' : 'energy-ml-test-card--normal'}`}>
                  <p className="energy-ml-test-card-status">
                    <strong>{severityLabel(mlTestResult.status, mlTestResult.severity)}</strong>
                  </p>
                  <div className="energy-ml-row"><span className="energy-ml-label">Power</span><span className="energy-ml-value">{Number(mlTestResult.power).toFixed(1)} W</span></div>
                  <div className="energy-ml-row"><span className="energy-ml-label">Energy</span><span className="energy-ml-value">{Number(mlTestResult.energy).toFixed(4)} kWh</span></div>
                  <div className="energy-ml-row"><span className="energy-ml-label">Recon. Error</span><span className="energy-ml-value">{Number(mlTestResult.reconstruction_error).toFixed(4)}</span></div>
                  <div className="energy-ml-row"><span className="energy-ml-label">Threshold</span><span className="energy-ml-value">{Number(mlTestResult.threshold).toFixed(4)}</span></div>
                  <div className="energy-ml-row"><span className="energy-ml-label">Deviation</span><span className="energy-ml-value">{Number(mlTestResult.deviation_percent).toFixed(1)}%</span></div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
      <p className="energy-ml-source">Data from API (GET /api/energy-data), updated every 2s.</p>
    </div>
  )
}

function VoltagePanel({ buildingState, voltageHistory, roomIds, roomTitles, normalMin, normalMax }) {
  return (
    <div className="tab-panel-content voltage-panel">
      <h3 className="tab-panel-title">Voltage monitoring</h3>
      <p className="tab-panel-text">Average voltage per room (from 3 nodes). Graph updates every simulation cycle. Normal range: {normalMin}–{normalMax} V.</p>
      <div className="voltage-room-list">
        {roomIds.map((roomId) => {
          const avg = getAverageVoltageForRoom(buildingState, roomId)
          const history = voltageHistory[roomId] || []
          const isAbnormal = avg < normalMin || avg > normalMax
          return (
            <div
              key={roomId}
              className={`voltage-room-card ${isAbnormal ? 'voltage-room-card--abnormal' : ''}`}
            >
              <div className="voltage-room-header">
                <span className="voltage-room-title">{roomTitles[roomId] ?? roomId}</span>
                <span className={`voltage-room-avg ${isAbnormal ? 'voltage-room-avg--abnormal' : ''}`}>
                  {avg.toFixed(1)} V
                </span>
              </div>
              <VoltageSparkline data={history} normalMin={normalMin} normalMax={normalMax} />
            </div>
          )
        })}
      </div>
    </div>
  )
}

function VoltageSparkline({ data, normalMin, normalMax }) {
  const canvasRef = useRef(null)
  const w = 220
  const h = 44

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !data.length) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, w, h)
    const min = Math.min(normalMin, ...data)
    const max = Math.max(normalMax, ...data)
    const range = max - min || 1
    const pad = 2
    const xScale = (w - 2 * pad) / Math.max(1, data.length - 1)
    const yScale = (h - 2 * pad) / range
    ctx.strokeStyle = '#f85149'
    ctx.lineWidth = 1.5
    ctx.beginPath()
    data.forEach((v, i) => {
      const x = pad + i * xScale
      const y = h - pad - (v - min) * yScale
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.stroke()
    ctx.strokeStyle = 'rgba(248, 81, 73, 0.5)'
    ctx.setLineDash([2, 2])
    const yMin = h - pad - (normalMin - min) * yScale
    const yMax = h - pad - (normalMax - min) * yScale
    ctx.beginPath()
    ctx.moveTo(pad, yMin)
    ctx.lineTo(w - pad, yMin)
    ctx.moveTo(pad, yMax)
    ctx.lineTo(w - pad, yMax)
    ctx.stroke()
    ctx.setLineDash([])
  }, [data, normalMin, normalMax])

  if (!data.length) return <div className="voltage-sparkline voltage-sparkline--empty">No data yet</div>
  return <canvas ref={canvasRef} width={w} height={h} className="voltage-sparkline" aria-label="Voltage history" />
}

function RoomPanel({ roomId, title, buildingState, variable }) {
  const values = getVariableValuesForRoom(buildingState, roomId, variable)
  const roomStatus = getRoomStatus(buildingState, roomId)
  return (
    <div className={`room-panel ${roomStatus === 'alert' ? 'room-panel--alert' : ''}`}>
      <h2 className="room-panel-title">{title}</h2>
      <div className="room-panel-heatmap">
        <FloorPlanWithCanvas sensorValues={values} variable={variable} />
      </div>
    </div>
  )
}

function FloorPlanWithCanvas({ sensorValues, variable }) {
  const canvasRef = useRef(null)
  const targetValues = Array.isArray(sensorValues) ? sensorValues.slice(0, 3) : [0, 0, 0]
  const [displayValues, setDisplayValues] = useState(() => targetValues.slice())
  const displayValuesRef = useRef(displayValues)
  const animationIdRef = useRef(null)
  const prevVariableRef = useRef(variable)

  displayValuesRef.current = displayValues

  // When target (sensorValues) or variable changes: animate over 500ms or snap
  useEffect(() => {
    if (prevVariableRef.current !== variable) {
      prevVariableRef.current = variable
      if (animationIdRef.current != null) cancelAnimationFrame(animationIdRef.current)
      setDisplayValues(targetValues.slice())
      displayValuesRef.current = targetValues.slice()
      return
    }

    const prev = displayValuesRef.current.slice()
    const target = targetValues.slice()
    const allSame = prev.length === target.length && prev.every((p, i) => Math.abs((Number(p) || 0) - (Number(target[i]) || 0)) < 1e-6)
    if (allSame) return

    if (animationIdRef.current != null) cancelAnimationFrame(animationIdRef.current)
    const startTime = Date.now()
    const duration = HEATMAP_TRANSITION_MS

    function tick() {
      const elapsed = Date.now() - startTime
      const progress = Math.min(1, elapsed / duration)
      const eased = easeOutCubic(progress)
      const interp = interpolateValues(prev, target, eased)
      setDisplayValues(interp)
      displayValuesRef.current = interp
      if (progress < 1) {
        animationIdRef.current = requestAnimationFrame(tick)
      } else {
        setDisplayValues(target)
        displayValuesRef.current = target
        animationIdRef.current = null
      }
    }
    animationIdRef.current = requestAnimationFrame(tick)
    return () => {
      if (animationIdRef.current != null) cancelAnimationFrame(animationIdRef.current)
    }
  }, [variable, targetValues[0], targetValues[1], targetValues[2]])

  const hasData = displayValues.some((v) => Number(v) > 0)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    if (!hasData) {
      ctx.clearRect(0, 0, CANVAS_W, CANVAS_H)
      return
    }
    drawHeatmap(ctx, displayValues, variable)
  }, [hasData, variable, displayValues[0], displayValues[1], displayValues[2]])

  return (
    <div className="floorplan-wrapper" style={{ position: 'relative', display: 'block', width: '100%', height: '100%', minHeight: 200 }}>
      <svg
        version="1.1"
        xmlns="http://www.w3.org/2000/svg"
        viewBox={`0 0 ${VIEWBOX.w} ${VIEWBOX.h}`}
        className="floorplan-svg"
        style={{ display: 'block', width: '100%', height: 'auto', verticalAlign: 'top' }}
      >
        <style>
          {`
            .st0{fill:#FFFFFF;stroke:#000000;stroke-miterlimit:10;}
            .st1{fill:#FFFFFF;stroke:#000000;stroke-width:3;stroke-miterlimit:10;}
            .st2{fill:none;}
            .st3{font-family:'SourceSansPro-Regular', sans-serif;}
            .st4{font-size:48px;}
            .st5{letter-spacing:4;}
          `}
        </style>
        <rect x="247.5" y="157.94" className="st0" width="1428" height="762" />
        <rect x="1037.5" y="157.94" className="st1" width="108" height="381" />
        <rect x="1146.5" y="157.94" className="st0" width="529" height="381" />
        <rect x="847.5" y="734.94" className="st0" width="828" height="185" />
        <rect x="383.65" y="670.79" className="st0" width="425" height="249.16" />
        <rect x="449.83" y="157.94" className="st0" width="452" height="358" />
        <rect x="1663.5" y="557.13" className="st0" width="12" height="161" />
        <rect
          x="1608.33"
          y="584.13"
          transform="matrix(0.7071 -0.7071 0.7071 0.7071 2.8637 1336.1719)"
          className="st0"
          width="12"
          height="161"
        />
        <rect x="247.5" y="246.85" className="st0" width="21.27" height="584.18" />
        <rect x="629.83" y="310.46" className="st2" width="92" height="52.96" />
        <text transform="matrix(1 0 0 1 629.8301 344.6381)" className="st3 st4 st5">BED</text>
        <rect x="524.15" y="764.37" className="st2" width="144" height="62" />
        <text transform="matrix(1 0 0 1 524.1503 798.5414)" className="st3 st4 st5">TABLE</text>
        <rect x="1281" y="311.7" className="st2" width="260" height="50.49" />
        <text transform="matrix(1 0 0 1 1280.9993 345.8745)" className="st3 st4 st5">RESTROOM</text>
        <rect x="1162.78" y="800.64" className="st2" width="236.44" height="53.62" />
        <text transform="matrix(1 0 0 1 1162.7802 834.8112)" className="st3 st4 st5">WARDROBE</text>
        <circle cx="275.3" cy="184" r="8" fill="none" stroke="#333" strokeWidth="2" />
        <circle cx="275.3" cy="900" r="8" fill="none" stroke="#333" strokeWidth="2" />
        <circle cx="1399.22" cy="625.63" r="8" fill="none" stroke="#333" strokeWidth="2" />
        <rect x="1062.5" y="314.44" className="st2" width="58" height="45" />
        <text transform="matrix(1 0 0 1 1062.4987 348.6193)" className="st3 st4 st5">AC</text>
      </svg>
      <canvas
        ref={canvasRef}
        width={CANVAS_W}
        height={CANVAS_H}
        className="heatmap-canvas"
        style={{
          position: 'absolute',
          left: 0,
          top: 0,
          width: '100%',
          height: '100%',
          pointerEvents: 'none',
        }}
        aria-hidden
      />
      {SENSORS.map((s, i) => {
        const raw = Number(displayValues[i]) || 0
        const norm = normalizeForColor(raw, variable)
        const bg = valueToColor(norm)
        const unit = variable === 'temperature' ? '°C' : variable === 'humidity' ? '%' : ''
        const label = variable === 'temperature' ? raw.toFixed(1) : Math.round(raw)
        const leftPct = ((s.x - VIEWBOX.w * 0) / VIEWBOX.w) * 100
        const topPct = ((s.y - VIEWBOX.h * 0) / VIEWBOX.h) * 100
        return (
          <span
            key={i}
            style={{
              position: 'absolute',
              left: `${leftPct}%`,
              top: `${topPct}%`,
              transform: 'translate(-50%, -50%)',
              background: bg,
              color: norm > 120 ? '#000' : '#fff',
              fontSize: '10px',
              fontWeight: 700,
              padding: '2px 5px',
              borderRadius: 4,
              border: '1px solid rgba(0,0,0,0.35)',
              pointerEvents: 'none',
              whiteSpace: 'nowrap',
              lineHeight: 1.2,
              zIndex: 2,
            }}
          >
            {label}{unit}
          </span>
        )
      })}
    </div>
  )
}
