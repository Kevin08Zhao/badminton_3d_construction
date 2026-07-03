import { useEffect, useMemo, useRef, useState } from 'react'
import { AnalysisExportPanel } from './components/AnalysisExportPanel'
import { KeyMetricsCard } from './components/KeyMetricsCard'
import { SettingsModal } from './components/SettingsModal'
import { ToastHost } from './components/ToastHost'
import { VideoStage } from './components/VideoStage'
import { WorkflowPanel } from './components/WorkflowPanel'
import {
  ApiHttpError,
  apiGetJob,
  apiGetLogs,
  apiStartJob,
  apiUploadVideo,
  resolvePublicUrl,
  type JobResponse,
  type PerfMode,
} from './api/client'

const RECENT_JOB_IDS_KEY = 'shuttlevision.recentJobIds'
const MAX_RECENT_JOB_IDS = 12

function App() {
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const [perfMode, setPerfMode] = useState<PerfMode>('standard')
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [videoObjectUrl, setVideoObjectUrl] = useState<string | null>(null)
  const [videoId, setVideoId] = useState<string | null>(null)
  const [calibrationPoints, setCalibrationPoints] = useState<
    { x: number; y: number; displayX: number; displayY: number }[]
  >([])

  const [jobId, setJobId] = useState<string | null>(null)
  const [job, setJob] = useState<JobResponse | null>(null)
  const [videoTimeSec, setVideoTimeSec] = useState(0)
  const [videoDurationSec, setVideoDurationSec] = useState(0)
  const [historyJobIdInput, setHistoryJobIdInput] = useState('')
  const [recentJobIds, setRecentJobIds] = useState<string[]>([])
  const [historyLoadPending, setHistoryLoadPending] = useState(false)
  const [userError, setUserError] = useState<string | null>(null)
  const [startPending, setStartPending] = useState(false)
  const [logLines, setLogLines] = useState<string[]>([])
  const logsOffsetRef = useRef(0)
  const pollTimerRef = useRef<number | null>(null)
  const pollFailuresRef = useRef(0)

  const mock = useMemo(
    () => ({
      workflow: {
        currentStep: 0,
        steps: [
          { title: '导入视频', subtitle: 'Import Video' },
          { title: '相机标定(6点)', subtitle: 'Calibration' },
          { title: 'AI 分析运行', subtitle: 'AI Analysis' },
          { title: '3D 轨迹重建', subtitle: '3D Reconstruction' },
          { title: '结果导出', subtitle: 'Export' },
        ],
      },
      settings: {
        perfMode: 'standard' as const,
        batchSize: 8,
        weights: [
          { label: 'TrackNet 权重', path: 'data/weights/ckpts/TrackNet_best.pt', ok: true },
          { label: 'HitNet 权重', path: 'data/weights/hitnet_output/hitnet_overfit_best.pth', ok: true },
          { label: 'YOLO Pose 权重', path: 'yolov8x-pose.pt', ok: true },
        ],
      },
    }),
    [],
  )

  const analysisReady = job?.status === 'succeeded'

  const trajectoryFpsHint = useMemo(() => {
    const v = job?.meta && typeof (job.meta as Record<string, unknown>).video_fps === 'number'
      ? (job.meta as { video_fps: number }).video_fps
      : null
    return v != null && Number.isFinite(v) && v > 0 ? v : null
  }, [job?.meta])

  const metricsFromJob = useMemo(() => {
    if (!analysisReady || !job?.meta) return null
    const meta = job.meta as Record<string, unknown>
    const shots = typeof meta.shots === 'number' ? meta.shots : null
    const num = (k: string) => (typeof meta[k] === 'number' ? (meta[k] as number) : null)
    return {
      detectionCount: shots,
      nearDistanceM: num('near_player_distance_m'),
      farDistanceM: num('far_player_distance_m'),
      nearAvgSpeedMps: num('near_player_avg_speed_mps'),
      farAvgSpeedMps: num('far_player_avg_speed_mps'),
    }
  }, [analysisReady, job?.meta])

  const resolvedArtifacts = useMemo(() => {
    if (!job?.artifacts) return null
    return Object.fromEntries(
      Object.entries(job.artifacts).map(([k, v]) => [k, resolvePublicUrl(v)]),
    )
  }, [job?.artifacts])

  const displayVideoUrl = useMemo(() => {
    if (!analysisReady) return videoObjectUrl
    return resolvedArtifacts?.overlay_ball ?? resolvedArtifacts?.mp4 ?? videoObjectUrl
  }, [analysisReady, resolvedArtifacts?.overlay_ball, resolvedArtifacts?.mp4, videoObjectUrl])

  const chartData = useMemo(() => {
    if (!analysisReady) return []
    return Array.from({ length: 46 }).map((_, i) => ({
      frame: i,
      z: Math.max(
        0.2,
        1.2 +
          1.8 * Math.sin((i / 45) * Math.PI) +
          0.15 * Math.cos((i / 12) * Math.PI),
      ),
      speed: Math.max(
        1,
        13 +
          8 * Math.sin((i / 45) * Math.PI) +
          2.5 * Math.cos((i / 10) * Math.PI),
      ),
    }))
  }, [analysisReady])

  const currentStep = useMemo(() => {
    if (job?.status === 'succeeded') return 4
    if (job?.status === 'running' || job?.status === 'queued') return 2
    if (!videoId) return 0
    if (calibrationPoints.length < 6) return 1
    if (!jobId) return 2
    return 2
  }, [videoId, calibrationPoints.length, jobId, job?.status])

  const progress = job?.progress ?? 0

  function rememberRecentJobId(id: string) {
    const clean = id.trim()
    if (!clean) return
    setRecentJobIds((prev) => {
      const next = [clean, ...prev.filter((x) => x !== clean)].slice(0, MAX_RECENT_JOB_IDS)
      try {
        window.localStorage.setItem(RECENT_JOB_IDS_KEY, JSON.stringify(next))
      } catch {
        // ignore localStorage failures
      }
      return next
    })
  }

  useEffect(() => {
    if (!videoFile) return
    const url = URL.createObjectURL(videoFile)
    setVideoObjectUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [videoFile])

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(RECENT_JOB_IDS_KEY)
      if (!raw) return
      const parsed = JSON.parse(raw)
      if (!Array.isArray(parsed)) return
      const next = parsed
        .filter((x): x is string => typeof x === 'string')
        .map((x) => x.trim())
        .filter(Boolean)
        .slice(0, MAX_RECENT_JOB_IDS)
      setRecentJobIds(next)
    } catch {
      // ignore invalid cache
    }
  }, [])

  useEffect(() => {
    if (!jobId) return
    logsOffsetRef.current = 0
    setLogLines([])
    pollFailuresRef.current = 0

    const tick = async () => {
      try {
        const j = await apiGetJob(jobId)
        pollFailuresRef.current = 0
        setJob(j)
        try {
          const logs = await apiGetLogs(jobId, logsOffsetRef.current)
          logsOffsetRef.current = logs.next_offset
          if (logs.lines.length) setLogLines((prev) => [...prev, ...logs.lines])
        } catch {
          /* logs are best-effort; job status still updates */
        }
        if (j.status === 'succeeded' || j.status === 'failed') {
          if (pollTimerRef.current) window.clearInterval(pollTimerRef.current)
          pollTimerRef.current = null
        }
      } catch (e) {
        if (e instanceof ApiHttpError && e.status === 404) {
          pollFailuresRef.current = 0
          if (pollTimerRef.current) window.clearInterval(pollTimerRef.current)
          pollTimerRef.current = null
          setJobId(null)
          setJob({
            id: jobId,
            status: 'failed',
            progress: 0,
            step: 'lost',
            error: '任务不存在（后端已重启或未找到记录）。请重新运行分析。',
          })
          return
        }
        pollFailuresRef.current += 1
        if (pollFailuresRef.current >= 5) {
          if (pollTimerRef.current) window.clearInterval(pollTimerRef.current)
          pollTimerRef.current = null
          const msg =
            e instanceof ApiHttpError
              ? e.message
              : e instanceof Error
                ? e.message
                : '请求失败'
          setJob({
            id: jobId,
            status: 'failed',
            progress: 0,
            step: 'poll_error',
            error: `${msg}（已多次连接失败并停止轮询；请启动后端后点击「开始分析」重试。）`,
          })
        }
      }
    }

    tick()
    pollTimerRef.current = window.setInterval(tick, 1200)
    return () => {
      if (pollTimerRef.current) window.clearInterval(pollTimerRef.current)
      pollTimerRef.current = null
    }
  }, [jobId])

  const toasts = useMemo(() => {
    const list: { kind: 'warning' | 'success' | 'info'; title: string; message: string }[] = []
    if (userError) {
      list.push({ kind: 'warning', title: '请求失败', message: userError })
    }
    if (job?.status === 'failed' && job.error) {
      list.push({ kind: 'warning', title: '任务失败', message: job.error })
    }
    const reproj = job?.meta?.reproj_error_px
    if (typeof reproj === 'number' && reproj >= 5) {
      list.push({
        kind: 'warning',
        title: '标定精度较低',
        message: `重投影误差为 ${reproj.toFixed(1)}px，建议重新标定`,
      })
    }
    return list
  }, [userError, job?.error, job?.meta?.reproj_error_px, job?.status])

  async function handleUpload(file: File) {
    setUserError(null)
    setVideoFile(file)
    setVideoId(null)
    setCalibrationPoints([])
    setJobId(null)
    setJob(null)
    setLogLines([])
    logsOffsetRef.current = 0

    try {
      const resp = await apiUploadVideo(file)
      setVideoId(resp.video_id)
    } catch (e) {
      setVideoFile(null)
      setVideoId(null)
      const msg =
        e instanceof ApiHttpError
          ? e.message.trim() || `HTTP ${e.status}`
          : e instanceof Error
            ? e.message
            : String(e)
      setUserError(msg || '上传失败')
    }
  }

  function handleAddCalibrationPointMapped(p: {
    x: number
    y: number
    displayX: number
    displayY: number
  }) {
    setCalibrationPoints((prev) => (prev.length >= 6 ? prev : [...prev, p]))
  }

  function handleUndoCalibration() {
    setCalibrationPoints((prev) => prev.slice(0, -1))
  }

  function handleResetCalibration() {
    setCalibrationPoints([])
  }

  async function handleStart() {
    if (!videoId) return
    if (calibrationPoints.length !== 6) return
    setUserError(null)
    setStartPending(true)
    try {
      const resp = await apiStartJob({
        video_id: videoId,
        perf_mode: perfMode,
        calibration_points: calibrationPoints.map((p) => ({ x: p.x, y: p.y })),
      })
      setJobId(resp.job_id)
      rememberRecentJobId(resp.job_id)
    } catch (e) {
      const msg =
        e instanceof ApiHttpError
          ? e.message.trim() || `HTTP ${e.status}`
          : e instanceof Error
            ? e.message
            : String(e)
      setUserError(msg || '无法开始分析')
    } finally {
      setStartPending(false)
    }
  }

  async function handleLoadHistoryJob() {
    const id = historyJobIdInput.trim()
    if (!id) {
      setUserError('请先输入历史任务 job_id')
      return
    }
    setUserError(null)
    setHistoryLoadPending(true)
    try {
      const j = await apiGetJob(id)
      // 不重跑模型：直接切换到指定历史任务并进入同一轮询/日志流程
      setJobId(id)
      setJob(j)
      rememberRecentJobId(id)
    } catch (e) {
      const msg =
        e instanceof ApiHttpError
          ? e.message.trim() || `HTTP ${e.status}`
          : e instanceof Error
            ? e.message
            : String(e)
      setUserError(msg || '无法加载历史任务')
    } finally {
      setHistoryLoadPending(false)
    }
  }

  return (
    <div className="h-screen w-screen p-4 flex gap-4 overflow-hidden box-border bg-slate-900">
      <div className="w-[20%] min-w-[260px] h-full flex flex-col gap-4 overflow-hidden">
        <WorkflowPanel
          workflow={{ ...mock.workflow, currentStep }}
          hasAnalysisData={analysisReady}
          trajectoryCsvUrl={resolvedArtifacts?.csv ?? null}
          trajectoryFpsHint={trajectoryFpsHint}
          currentTimeSec={videoTimeSec}
          videoDurationSec={videoDurationSec}
        />
      </div>

      <div className="w-[50%] min-w-0 h-full flex flex-col gap-4 overflow-hidden">
        <div className="bg-slate-800 border border-slate-700 rounded-xl p-3 flex items-center gap-2">
          <div className="text-xs text-slate-300 shrink-0">加载历史任务</div>
          <select
            value=""
            onChange={(e) => {
              const v = e.target.value
              if (!v) return
              setHistoryJobIdInput(v)
            }}
            className="w-[200px] shrink-0 rounded-lg border border-slate-600 bg-slate-900 px-2 py-1.5 text-xs text-slate-100"
            title="最近任务"
          >
            <option value="">最近任务</option>
            {recentJobIds.map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))}
          </select>
          <input
            value={historyJobIdInput}
            onChange={(e) => setHistoryJobIdInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') void handleLoadHistoryJob()
            }}
            placeholder="输入 job_id（无需重跑模型）"
            className="min-w-0 flex-1 rounded-lg border border-slate-600 bg-slate-900 px-2 py-1.5 text-xs text-slate-100 placeholder:text-slate-500"
          />
          <button
            type="button"
            onClick={() => void handleLoadHistoryJob()}
            disabled={historyLoadPending}
            className="px-3 py-1.5 rounded-lg border border-slate-600 text-xs text-slate-100 bg-slate-700/70 hover:bg-slate-600/90 disabled:opacity-50"
          >
            {historyLoadPending ? '加载中…' : '加载'}
          </button>
        </div>
        <VideoStage
          currentStep={currentStep}
          onOpenSettings={() => setIsSettingsOpen(true)}
          videoObjectUrl={displayVideoUrl}
          calibrationPoints={calibrationPoints}
          onPickVideo={handleUpload}
          onAddCalibrationPoint={handleAddCalibrationPointMapped}
          onUndoCalibration={handleUndoCalibration}
          onResetCalibration={handleResetCalibration}
          onStartAnalysis={handleStart}
          progress={progress}
          logs={logLines}
          jobStatus={job?.status ?? null}
          startPending={startPending}
          onVideoTimeChange={setVideoTimeSec}
          onVideoDurationChange={setVideoDurationSec}
        />
      </div>

      <div className="w-[30%] min-w-[320px] h-full flex flex-col gap-2 overflow-hidden min-h-0">
        <KeyMetricsCard metrics={metricsFromJob} />
        <div className="flex-1 min-h-0 flex flex-col">
          <AnalysisExportPanel
            data={chartData}
            hasData={analysisReady}
            artifacts={resolvedArtifacts}
            disabled={job?.status !== 'succeeded'}
            className="h-full"
          />
        </div>
      </div>

      <SettingsModal
        open={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        defaultMode={perfMode}
        batchSize={mock.settings.batchSize}
        weights={mock.settings.weights}
        onChangeMode={setPerfMode}
      />
      <ToastHost toasts={toasts} />
    </div>
  )
}

export default App
