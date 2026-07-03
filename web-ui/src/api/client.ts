export class ApiHttpError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiHttpError'
    this.status = status
  }
}

function formatHttpErrorMessage(status: number, rawBody: string): string {
  const t = rawBody.trim()
  const lowered = t.toLowerCase()
  if (
    [502, 503, 504].includes(status) &&
    (lowered.includes('econnrefused') ||
      lowered.includes('econnreset') ||
      lowered.includes('connect') ||
      lowered.includes('proxy error'))
  ) {
    return '无法连接到后端 API。请在项目根目录启动：uvicorn backend.app:app --host 127.0.0.1 --port 8000'
  }
  if (t.length > 400) return `${t.slice(0, 400)}…`
  return t || `HTTP ${status}`
}

async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(apiUrl(path), init)
  } catch {
    throw new ApiHttpError(
      '网络错误：无法访问 API（本地开发请确认后端已在 127.0.0.1:8000 运行，且 Vite 代理未改端口）。',
      0,
    )
  }
}

async function throwIfNotOk(res: Response): Promise<void> {
  if (res.ok) return
  const raw = await res.text()
  throw new ApiHttpError(formatHttpErrorMessage(res.status, raw), res.status)
}

const API_ORIGIN = import.meta.env.VITE_API_BASE_URL?.trim()

/** Same-origin `/api` in dev/preview (proxied to backend); set `VITE_API_BASE_URL` when the UI is hosted separately. */
function apiUrl(path: string): string {
  const p = path.startsWith('/') ? path : `/${path}`
  if (!API_ORIGIN) return p
  return `${API_ORIGIN.replace(/\/$/, '')}${p}`
}

/** Backend returns `/api/artifacts/...` paths; prefix when the UI is not same-origin as the API. */
export function resolvePublicUrl(path: string): string {
  if (!API_ORIGIN || !path.startsWith('/')) return path
  return `${API_ORIGIN.replace(/\/$/, '')}${path}`
}

export type UploadResponse = { video_id: string; filename: string; path: string }

export type JobStatus =
  | 'queued'
  | 'running'
  | 'succeeded'
  | 'failed'

export type PerfMode = 'fast' | 'standard' | 'precise'

export type JobResponse = {
  id: string
  status: JobStatus
  progress: number
  step: string
  error?: string | null
  meta?: any
  artifacts?: Record<string, string>
}

export type LogsResponse = {
  offset: number
  next_offset: number
  lines: string[]
}

export async function apiUploadVideo(file: File): Promise<UploadResponse> {
  const fd = new FormData()
  fd.append('file', file)
  const res = await apiFetch('/api/upload', { method: 'POST', body: fd })
  await throwIfNotOk(res)
  return await res.json()
}

export async function apiStartJob(args: {
  video_id: string
  perf_mode: PerfMode
  calibration_points: { x: number; y: number }[]
}): Promise<{ job_id: string }> {
  const res = await apiFetch('/api/jobs', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(args),
  })
  await throwIfNotOk(res)
  return await res.json()
}

export async function apiGetJob(jobId: string): Promise<JobResponse> {
  const res = await apiFetch(`/api/jobs/${encodeURIComponent(jobId)}`)
  await throwIfNotOk(res)
  return await res.json()
}

export async function apiGetLogs(jobId: string, offset: number): Promise<LogsResponse> {
  const q = `offset=${offset}&limit=200`
  const res = await apiFetch(`/api/jobs/${encodeURIComponent(jobId)}/logs?${q}`)
  await throwIfNotOk(res)
  return await res.json()
}

