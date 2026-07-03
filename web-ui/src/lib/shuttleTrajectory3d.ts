/**
 * 解析 output_reconstructed_3d.csv，并按击球分段；提供将世界坐标投影到画布并绘制 3D 轨迹的工具。
 */

export type TrajectoryPoint = { frame: number; x: number; y: number; z: number }

export type ShotTrajectorySegment = {
  key: string
  rallyId: number
  shotNumber: number
  points: TrajectoryPoint[]
}

function parseCsvLine(line: string): string[] {
  const out: string[] = []
  let cur = ''
  let q = false
  for (let i = 0; i < line.length; i++) {
    const c = line[i]
    if (c === '"') {
      q = !q
      continue
    }
    if (!q && c === ',') {
      out.push(cur)
      cur = ''
      continue
    }
    cur += c
  }
  out.push(cur)
  return out
}

/**
 * 将 CSV 文本解析为按 (rally_id, shot_number) 分组的轨迹点（每段内按 frame 升序）。
 */
export function parseReconstructedTrajectoryCsv(csvText: string): ShotTrajectorySegment[] {
  const lines = csvText.split(/\r?\n/).filter((l) => l.trim().length > 0)
  if (lines.length < 2) return []

  const header = parseCsvLine(lines[0]).map((h) => h.trim().toLowerCase())
  const ri = header.indexOf('rally_id')
  const si = header.indexOf('shot_number')
  const fi = header.indexOf('frame')
  const xi = header.indexOf('x')
  const yi = header.indexOf('y')
  const zi = header.indexOf('z')
  if (ri < 0 || si < 0 || fi < 0 || xi < 0 || yi < 0 || zi < 0) return []
  const minCells = Math.max(ri, si, fi, xi, yi, zi) + 1

  const bucket = new Map<string, TrajectoryPoint[]>()

  for (let k = 1; k < lines.length; k++) {
    const cells = parseCsvLine(lines[k])
    if (cells.length < minCells) continue
    const rallyId = Number(cells[ri])
    const shotNumber = Number(cells[si])
    const frame = Number(cells[fi])
    const x = Number(cells[xi])
    const y = Number(cells[yi])
    const z = Number(cells[zi])
    if (![rallyId, shotNumber, frame, x, y, z].every((n) => Number.isFinite(n))) continue
    const key = `${rallyId}-${shotNumber}`
    if (!bucket.has(key)) bucket.set(key, [])
    bucket.get(key)!.push({ frame, x, y, z })
  }

  const segments: ShotTrajectorySegment[] = []
  for (const [key, pts] of bucket) {
    pts.sort((a, b) => a.frame - b.frame)
    const [rallyId, shotNumber] = key.split('-').map(Number)
    segments.push({ key, rallyId, shotNumber, points: pts })
  }
  segments.sort((a, b) => {
    if (a.rallyId !== b.rallyId) return a.rallyId - b.rallyId
    return a.shotNumber - b.shotNumber
  })
  return segments
}

export function maxTrajectoryFrame(segments: ShotTrajectorySegment[]): number {
  let m = 0
  for (const s of segments) {
    for (const p of s.points) m = Math.max(m, p.frame)
  }
  return m
}

type BBox = { minX: number; maxX: number; minY: number; maxY: number; minZ: number; maxZ: number }

function bboxOfSegments(segments: ShotTrajectorySegment[]): BBox | null {
  let minX = Infinity,
    maxX = -Infinity,
    minY = Infinity,
    maxY = -Infinity,
    minZ = Infinity,
    maxZ = -Infinity
  for (const s of segments) {
    for (const p of s.points) {
      minX = Math.min(minX, p.x)
      maxX = Math.max(maxX, p.x)
      minY = Math.min(minY, p.y)
      maxY = Math.max(maxY, p.y)
      minZ = Math.min(minZ, p.z)
      maxZ = Math.max(maxZ, p.z)
    }
  }
  if (!Number.isFinite(minX)) return null
  return { minX, maxX, minY, maxY, minZ, maxZ }
}

/** 世界系：X、Y 为地面，Z 向上。先绕 Z 转 yaw，再绕 X 转 pitch，再正交投影到画布平面（横轴 x'，纵轴 -z'）。 */
function worldToCanvas(
  x: number,
  y: number,
  z: number,
  yaw: number,
  pitch: number,
  cx: number,
  cy: number,
  scale: number,
): [number, number] {
  const c1 = Math.cos(yaw)
  const s1 = Math.sin(yaw)
  const x1 = c1 * x - s1 * y
  const y1 = s1 * x + c1 * y
  const z1 = z
  const c2 = Math.cos(pitch)
  const s2 = Math.sin(pitch)
  const x2 = x1
  const y2 = c2 * y1 - s2 * z1
  const z2 = s2 * y1 + c2 * z1
  void y2
  return [cx + scale * x2, cy - scale * z2]
}

function segmentColor(index: number): string {
  const h = (index * 47) % 360
  return `hsl(${h} 78% 58%)`
}

export type DrawShuttleTrajectory3DOptions = {
  /** 当前视频帧（含），仅绘制 frame ≤ currentFrame 的已发生轨迹 */
  currentFrame: number
  /** 未发生部分是否以低透明度显示（便于理解完整球路） */
  showFutureGhost?: boolean
  /** 视角：绕 Z（竖直）弧度 */
  yawRad?: number
  /** 视角：绕 X 弧度（俯视） */
  pitchRad?: number
}

/**
 * 在 2D Canvas 上以 3D 投影视角绘制羽毛球轨迹（多段击球不同颜色），并绘制简易 XYZ 坐标轴。
 */
export function drawShuttleTrajectory3D(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  segments: ShotTrajectorySegment[],
  opts: DrawShuttleTrajectory3DOptions,
): void {
  const padding = 10
  const yaw = opts.yawRad ?? 0.55
  const pitch = opts.pitchRad ?? -0.45
  const showGhost = opts.showFutureGhost ?? true
  const cf = Math.max(0, Math.floor(opts.currentFrame))

  ctx.clearRect(0, 0, width, height)

  const bbox = bboxOfSegments(segments)
  if (!bbox || segments.length === 0) {
    ctx.fillStyle = 'rgba(15, 23, 42, 0.92)'
    ctx.fillRect(0, 0, width, height)
    ctx.fillStyle = '#64748b'
    ctx.font = '11px ui-sans-serif, system-ui, sans-serif'
    ctx.fillText('无轨迹数据', padding, height / 2)
    return
  }

  const cxW = (bbox.minX + bbox.maxX) / 2
  const cyW = (bbox.minY + bbox.maxY) / 2
  const czW = (bbox.minZ + bbox.maxZ) / 2

  const corners: [number, number, number][] = [
    [bbox.minX, bbox.minY, bbox.minZ],
    [bbox.maxX, bbox.minY, bbox.minZ],
    [bbox.minX, bbox.maxY, bbox.minZ],
    [bbox.maxX, bbox.maxY, bbox.minZ],
    [bbox.minX, bbox.minY, bbox.maxZ],
    [bbox.maxX, bbox.minY, bbox.maxZ],
    [bbox.minX, bbox.maxY, bbox.maxZ],
    [bbox.maxX, bbox.maxY, bbox.maxZ],
  ]

  let minSx = Infinity,
    maxSx = -Infinity,
    minSy = Infinity,
    maxSy = -Infinity
  for (const [x, y, z] of corners) {
    const [sx, sy] = worldToCanvas(x - cxW, y - cyW, z - czW, yaw, pitch, 0, 0, 1)
    minSx = Math.min(minSx, sx)
    maxSx = Math.max(maxSx, sx)
    minSy = Math.min(minSy, sy)
    maxSy = Math.max(maxSy, sy)
  }

  const spanX = Math.max(maxSx - minSx, 1e-6)
  const spanY = Math.max(maxSy - minSy, 1e-6)
  const innerW = width - padding * 2
  const innerH = height - padding * 2
  const scale = Math.min(innerW / spanX, innerH / spanY) * 0.88
  const canvasCx = width / 2
  const canvasCy = height / 2

  const project = (x: number, y: number, z: number) =>
    worldToCanvas(x - cxW, y - cyW, z - czW, yaw, pitch, canvasCx, canvasCy, scale)

  ctx.fillStyle = 'rgba(15, 23, 42, 0.88)'
  ctx.fillRect(0, 0, width, height)
  ctx.strokeStyle = 'rgba(51, 65, 85, 0.9)'
  ctx.lineWidth = 1
  ctx.strokeRect(0.5, 0.5, width - 1, height - 1)

  const axisLen = Math.max(bbox.maxX - bbox.minX, bbox.maxY - bbox.minY, bbox.maxZ - bbox.minZ, 1) * 0.22
  const ox = bbox.minX
  const oy = bbox.minY
  const oz = bbox.minZ
  const axes: { from: [number, number, number]; to: [number, number, number]; color: string; label: string }[] = [
    { from: [ox, oy, oz], to: [ox + axisLen, oy, oz], color: '#f87171', label: 'X' },
    { from: [ox, oy, oz], to: [ox, oy + axisLen, oz], color: '#4ade80', label: 'Y' },
    { from: [ox, oy, oz], to: [ox, oy, oz + axisLen], color: '#60a5fa', label: 'Z' },
  ]
  for (const a of axes) {
    const [sx0, sy0] = project(...a.from)
    const [sx1, sy1] = project(...a.to)
    ctx.strokeStyle = a.color
    ctx.lineWidth = 1.5
    ctx.beginPath()
    ctx.moveTo(sx0, sy0)
    ctx.lineTo(sx1, sy1)
    ctx.stroke()
    ctx.fillStyle = a.color
    ctx.font = 'bold 9px ui-sans-serif, system-ui, sans-serif'
    ctx.fillText(a.label, sx1 + 3, sy1 + 3)
  }

  const drawSegment = (pts: TrajectoryPoint[], color: string, alpha: number) => {
    if (pts.length < 2) return
    ctx.strokeStyle = color
    ctx.globalAlpha = alpha
    ctx.lineWidth = 2
    ctx.lineJoin = 'round'
    ctx.lineCap = 'round'
    ctx.beginPath()
    const [sx0, sy0] = project(pts[0].x, pts[0].y, pts[0].z)
    ctx.moveTo(sx0, sy0)
    for (let i = 1; i < pts.length; i++) {
      const [sx, sy] = project(pts[i].x, pts[i].y, pts[i].z)
      ctx.lineTo(sx, sy)
    }
    ctx.stroke()
    ctx.globalAlpha = 1
  }

  segments.forEach((seg, idx) => {
    const color = segmentColor(idx)
    const past = seg.points.filter((p) => p.frame <= cf)
    if (showGhost && seg.points.length > past.length) {
      drawSegment(seg.points, '#475569', 0.25)
    }
    drawSegment(past, color, 0.95)
    const last = past.length ? past[past.length - 1] : null
    if (last) {
      const [sx, sy] = project(last.x, last.y, last.z)
      ctx.fillStyle = '#fbbf24'
      ctx.beginPath()
      ctx.arc(sx, sy, 3.5, 0, Math.PI * 2)
      ctx.fill()
    }
  })

  ctx.fillStyle = '#94a3b8'
  ctx.font = '10px ui-mono, ui-monospace, monospace'
  ctx.fillText(`f ≤ ${cf}`, padding, height - padding)
}

/**
 * 由视频时间与帧率得到当前帧索引（与 CSV 中 frame 列对齐）。
 */
export function videoTimeToFrameIndex(timeSec: number, fps: number): number {
  if (!Number.isFinite(timeSec) || !Number.isFinite(fps) || fps <= 0) return 0
  return Math.max(0, Math.floor(timeSec * fps + 1e-9))
}
