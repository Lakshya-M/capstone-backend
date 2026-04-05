/**
 * Central state for the smart building digital twin.
 * buildingState.building.rooms[roomId].nodes
 */

const ROOM_IDS = ['roomA', 'roomB', 'roomC', 'roomD']

function createNode(id, hardware = false, seed = 0) {
  const s = (id.charCodeAt(0) + (id.charCodeAt(1) || 0) + seed) % 200
  return {
    id,
    aqi: (s % 160) + 30,
    temperature: 18 + (s % 22),
    humidity: 25 + (s % 55),
    voltage: 220 + (s % 15),
    status: 'normal',
    hardware,
  }
}

function createRoom(roomId, nodeIds) {
  const seed = roomId.split('').reduce((a, c) => a + c.charCodeAt(0), 0)
  return {
    status: 'normal',
    nodes: Object.fromEntries(
      nodeIds.map((nid, i) => [nid, createNode(nid, nid === 'D1', seed + i)])
    ),
  }
}

export function getInitialBuildingState() {
  return {
    building: {
      rooms: {
        roomA: createRoom('roomA', ['A1', 'A2', 'A3']),
        roomB: createRoom('roomB', ['B1', 'B2', 'B3']),
        roomC: createRoom('roomC', ['C1', 'C2', 'C3']),
        roomD: createRoom('roomD', ['D1', 'D2', 'D3']),
      },
    },
  }
}

export const ROOM_IDS_LIST = ROOM_IDS

export function getNodesForRoom(buildingState, roomId) {
  const room = buildingState?.building?.rooms?.[roomId]
  if (!room?.nodes) return []
  return Object.values(room.nodes).sort((a, b) => a.id.localeCompare(b.id))
}

export function getRoomStatus(buildingState, roomId) {
  const room = buildingState?.building?.rooms?.[roomId]
  return room?.status ?? 'normal'
}

export function getAqiValuesForRoom(buildingState, roomId) {
  return getNodesForRoom(buildingState, roomId).map((n) => Number(n.aqi) || 0)
}

/** Get 3 node values for the given variable (aqi | temperature | humidity) for a room */
export function getVariableValuesForRoom(buildingState, roomId, variable) {
  const nodes = getNodesForRoom(buildingState, roomId)
  if (variable === 'aqi') return nodes.map((n) => Number(n.aqi) || 0)
  if (variable === 'temperature') return nodes.map((n) => Number(n.temperature) || 0)
  if (variable === 'humidity') return nodes.map((n) => Number(n.humidity) || 0)
  return nodes.map((n) => Number(n.aqi) || 0)
}

/** Average voltage across the 3 nodes in a room */
export function getAverageVoltageForRoom(buildingState, roomId) {
  const nodes = getNodesForRoom(buildingState, roomId)
  if (!nodes.length) return 0
  const sum = nodes.reduce((s, n) => s + (Number(n.voltage) || 0), 0)
  return Number((sum / nodes.length).toFixed(1))
}

export function updateNodeInBuildingState(prev, roomId, nodeId, updates) {
  const next = JSON.parse(JSON.stringify(prev))
  const node = next.building?.rooms?.[roomId]?.nodes?.[nodeId]
  if (!node) return prev
  Object.assign(node, updates)
  return next
}

export function updateRoomNodesInBuildingState(prev, roomId, nodeUpdates) {
  const next = JSON.parse(JSON.stringify(prev))
  const room = next.building?.rooms?.[roomId]
  if (!room) return prev
  for (const [nodeId, updates] of Object.entries(nodeUpdates)) {
    if (room.nodes[nodeId]) Object.assign(room.nodes[nodeId], updates)
  }
  return next
}
