/**
 * Sensor simulation engine.
 * Every cycle: adjust non-hardware node values by small deltas for gradual, realistic change.
 */

import { ROOM_IDS_LIST } from './buildingState'

/** Clamp ranges — wide enough for visible heatmap colour shifts */
const CLAMP = {
  aqi: { min: 10, max: 300 },
  temperature: { min: 15, max: 45 },
  humidity: { min: 20, max: 95 },
  voltage: { min: 210, max: 240 },
}

/** Max absolute change per cycle (symmetric ±) — large enough to see on every tick */
const DELTA = {
  aqi: 18,
  temperature: 2.5,
  humidity: 6,
  voltage: 2,
}

/** Random value in [-range, +range] */
function randomDelta(range) {
  return (Math.random() * 2 - 1) * range
}

function randomInRange(min, max) {
  return min + Math.random() * (max - min)
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, Number(value) || 0))
}

/** Per-room baseline drift (applied every cycle); nudge every 30s */
const TREND_RANGES = {
  aqi: { min: -4, max: 4 },
  temp: { min: -0.8, max: 0.8 },
  humidity: { min: -1.5, max: 1.5 },
}
const TREND_NUDGE = { aqi: 1.0, temp: 0.2, humidity: 0.4 }

let roomTrends = null

function getRoomTrends() {
  if (!roomTrends) {
    roomTrends = {}
    for (const roomId of ROOM_IDS_LIST) {
      roomTrends[roomId] = {
        aqiTrend: randomInRange(TREND_RANGES.aqi.min, TREND_RANGES.aqi.max),
        tempTrend: randomInRange(TREND_RANGES.temp.min, TREND_RANGES.temp.max),
        humidityTrend: randomInRange(TREND_RANGES.humidity.min, TREND_RANGES.humidity.max),
      }
    }
  }
  return roomTrends
}

/** Slightly change trend values every 30s for natural drift evolution */
export function nudgeRoomTrends() {
  const trends = getRoomTrends()
  for (const roomId of ROOM_IDS_LIST) {
    const t = trends[roomId]
    t.aqiTrend = clamp(t.aqiTrend + randomDelta(TREND_NUDGE.aqi), TREND_RANGES.aqi.min, TREND_RANGES.aqi.max)
    t.tempTrend = clamp(t.tempTrend + randomDelta(TREND_NUDGE.temp), TREND_RANGES.temp.min, TREND_RANGES.temp.max)
    t.humidityTrend = clamp(t.humidityTrend + randomDelta(TREND_NUDGE.humidity), TREND_RANGES.humidity.min, TREND_RANGES.humidity.max)
  }
}

/** Apply one cycle: previous value + small delta, then clamp */
function applyDelta(prev, deltaRange, clampMin, clampMax, round = false) {
  const current = Number(prev) || 0
  const next = clamp(current + randomDelta(deltaRange), clampMin, clampMax)
  return round ? Math.round(next) : Number(next.toFixed(1))
}

/** Anomaly rules: run after every sensor update */
const ANOMALY = {
  aqiHigh: 150,        // AQI > 150 → Air Quality Alert
  tempHigh: 32,        // Temperature > 32 → Temperature Alert
  humidityHigh: 75,    // Humidity > 75 → Humidity Alert
  voltageLow: 200,     // Voltage < 200 or > 250 → Power Alert
  voltageHigh: 250,
}

export function checkAnomaly(node) {
  const aqi = Number(node.aqi) || 0
  const temp = Number(node.temperature) || 0
  const humidity = Number(node.humidity) || 0
  const voltage = Number(node.voltage) || 0
  if (aqi > ANOMALY.aqiHigh) return { type: 'air_quality_alert', message: `Air Quality Alert: AQI ${aqi} (>${ANOMALY.aqiHigh})` }
  if (temp > ANOMALY.tempHigh) return { type: 'temperature_alert', message: `Temperature Alert: ${temp}°C (>${ANOMALY.tempHigh}°C)` }
  if (humidity > ANOMALY.humidityHigh) return { type: 'humidity_alert', message: `Humidity Alert: ${humidity}% (>${ANOMALY.humidityHigh}%)` }
  if (voltage < ANOMALY.voltageLow || voltage > ANOMALY.voltageHigh) return { type: 'power_alert', message: `Power Alert: Voltage ${voltage}V (valid 200-250V)` }
  return null
}

/**
 * Run one simulation cycle:
 * - Update all non-hardware nodes with new simulated readings
 * - Run anomaly detection and set node.status
 * - Return { nextState, alerts }
 */
export function runSimulationCycle(prevState) {
  const next = JSON.parse(JSON.stringify(prevState))
  const alerts = []

  const trends = getRoomTrends()

  for (const roomId of ROOM_IDS_LIST) {
    const room = next.building?.rooms?.[roomId]
    if (!room?.nodes) continue
    const roomTrend = trends[roomId]
    let roomHasAlert = false

    for (const [nodeId, node] of Object.entries(room.nodes)) {
      if (node.hardware) continue

      node.aqi = applyDelta(node.aqi, DELTA.aqi, CLAMP.aqi.min, CLAMP.aqi.max, true)
      node.aqi = clamp(node.aqi + roomTrend.aqiTrend, CLAMP.aqi.min, CLAMP.aqi.max)
      node.aqi = Math.round(node.aqi)

      node.temperature = applyDelta(node.temperature, DELTA.temperature, CLAMP.temperature.min, CLAMP.temperature.max)
      node.temperature = Number(clamp(node.temperature + roomTrend.tempTrend, CLAMP.temperature.min, CLAMP.temperature.max).toFixed(1))

      node.humidity = applyDelta(node.humidity, DELTA.humidity, CLAMP.humidity.min, CLAMP.humidity.max, true)
      node.humidity = Math.round(clamp(node.humidity + roomTrend.humidityTrend, CLAMP.humidity.min, CLAMP.humidity.max))

      node.voltage = applyDelta(node.voltage, DELTA.voltage, CLAMP.voltage.min, CLAMP.voltage.max)

      const anomaly = checkAnomaly(node)
      if (anomaly) {
        node.status = 'alert'
        roomHasAlert = true
        alerts.push({ roomId, nodeId, ...anomaly, timestamp: Date.now() })
      } else {
        node.status = 'normal'
      }
    }

    room.status = roomHasAlert ? 'alert' : 'normal'
  }

  // Hardware nodes skip simulated deltas; still evaluate alerts from their current values.
  for (const roomId of ROOM_IDS_LIST) {
    const room = next.building?.rooms?.[roomId]
    if (!room?.nodes) continue
    for (const node of Object.values(room.nodes)) {
      if (!node.hardware) continue
      const anomaly = checkAnomaly(node)
      if (anomaly) {
        node.status = 'alert'
        alerts.push({ roomId, nodeId: node.id, ...anomaly, timestamp: Date.now() })
      } else {
        node.status = 'normal'
      }
    }
    const anyAlert = Object.values(room.nodes).some((n) => n.status === 'alert')
    room.status = anyAlert ? 'alert' : 'normal'
  }

  return { nextState: next, alerts }
}

export const SIMULATION_INTERVAL_MS = 2000
export const TREND_NUDGE_INTERVAL_MS = 30000
export const PLAYBACK_SNAPSHOT_MAX = 300
