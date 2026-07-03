import { useEffect, useMemo, useRef, useState } from 'react'
import {
  ChevronDown,
  ChevronUp,
  Pause,
  Play,
  RotateCcw,
  Settings,
  Undo2,
  Upload,
} from 'lucide-react'
import { Button, IconButton } from './ui'

type CalibPoint = { x: number; y: number; displayX: number; displayY: number }

const CALIBRATION_PROMPTS = [
  '远端左角',
  '远端右角',
  '近端左角',
  '近端右角',
  '左网柱',
  '右网柱',
]

export function VideoStage(props: {
  currentStep: number
  onOpenSettings: () => void
  videoObjectUrl: string | null
  calibrationPoints: CalibPoint[]
  onPickVideo: (file: File) => void
  onAddCalibrationPoint: (p: CalibPoint) => void
  onUndoCalibration: () => void
  onResetCalibration: () => void
  onStartAnalysis: () => void | Promise<void>
  progress: number
  logs: string[]
  jobStatus: 'queued' | 'running' | 'succeeded' | 'failed' | null
  startPending?: boolean
  /** 向父层同步视频当前播放时间（秒） */
  onVideoTimeChange?: (timeSec: number) => void
  /** 向父层同步视频总时长（秒） */
  onVideoDurationChange?: (durationSec: number) => void
}) {
  const isCalibration = props.currentStep === 1
  const hasVideo = Boolean(props.videoObjectUrl)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const [isLogsOpen, setIsLogsOpen] = useState(true)
  const [videoDuration, setVideoDuration] = useState(0)
  const [videoTime, setVideoTime] = useState(0)
  const [videoPlaying, setVideoPlaying] = useState(false)

  useEffect(() => {
    setVideoDuration(0)
    setVideoTime(0)
    setVideoPlaying(false)
    props.onVideoDurationChange?.(0)
    props.onVideoTimeChange?.(0)
  }, [props.videoObjectUrl])

  function formatClock(seconds: number) {
    if (!Number.isFinite(seconds)) return '0:00'
    const s = Math.max(0, Math.floor(seconds))
    const m = Math.floor(s / 60)
    const r = s % 60
    return `${m}:${r.toString().padStart(2, '0')}`
  }

  async function togglePlayPause() {
    const v = videoRef.current
    if (!v) return
    if (v.paused) {
      await v.play()
      setVideoPlaying(true)
    } else {
      v.pause()
      setVideoPlaying(false)
    }
  }

  function seekTo(t: number) {
    const v = videoRef.current
    if (!v || !Number.isFinite(t)) return
    const next = Math.max(0, Math.min(v.duration || 0, t))
    v.currentTime = next
    setVideoTime(next)
  }

  const nextCalibrationLabel = useMemo(() => {
    const idx = Math.min(props.calibrationPoints.length, 5)
    return CALIBRATION_PROMPTS[idx]
  }, [props.calibrationPoints.length])

  const calibrationHint = useMemo(() => {
    const i = Math.min(props.calibrationPoints.length + 1, 6)
    return `请点击画面【${nextCalibrationLabel}】(${i}/6)`
  }, [props.calibrationPoints.length, nextCalibrationLabel])

  const jobBusy = props.jobStatus === 'queued' || props.jobStatus === 'running'
  const canStart =
    hasVideo &&
    props.calibrationPoints.length === 6 &&
    !jobBusy &&
    !props.startPending

  function openPicker() {
    fileInputRef.current?.click()
  }

  function onPick(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]
    if (f) props.onPickVideo(f)
    e.target.value = ''
  }

  function handleCanvasClick(e: React.MouseEvent<HTMLDivElement>) {
    if (!isCalibration) return
    if (!hasVideo) return
    if (props.calibrationPoints.length >= 6) return
    const vw = videoRef.current?.videoWidth ?? 0
    const vh = videoRef.current?.videoHeight ?? 0
    if (!vw || !vh) return

    const rect = e.currentTarget.getBoundingClientRect()
    const clickX = e.clientX - rect.left
    const clickY = e.clientY - rect.top

    // object-contain mapping
    const scale = Math.min(rect.width / vw, rect.height / vh)
    const renderedW = vw * scale
    const renderedH = vh * scale
    const offX = (rect.width - renderedW) / 2
    const offY = (rect.height - renderedH) / 2

    // ignore clicks on letterbox area
    if (clickX < offX || clickX > offX + renderedW || clickY < offY || clickY > offY + renderedH) {
      return
    }

    const u = Math.round((clickX - offX) / scale)
    const v = Math.round((clickY - offY) / scale)
    const clampedU = Math.max(0, Math.min(vw - 1, u))
    const clampedV = Math.max(0, Math.min(vh - 1, v))

    props.onAddCalibrationPoint({
      x: clampedU,
      y: clampedV,
      displayX: Math.round(clickX),
      displayY: Math.round(clickY),
    })
  }

  return (
    <div className="h-full flex flex-col gap-4 overflow-hidden">
      <div className="bg-slate-800 border border-slate-700 rounded-xl p-3 flex items-center justify-between">
        <div className="flex items-center gap-2 min-w-0">
          <Button variant="primary" onClick={openPicker}>
            <Upload className="h-4 w-4" />
            导入视频
          </Button>
          <div className="text-xs text-slate-400 truncate">
            建议：1080P 固定机位，完整覆盖球场
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            className="hidden"
            onChange={onPick}
          />
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            className="py-2"
            onClick={() => void props.onStartAnalysis()}
            disabled={!canStart}
          >
            开始分析
          </Button>
          <IconButton label="高级设置" onClick={props.onOpenSettings}>
            <Settings className="h-5 w-5 text-slate-200" />
          </IconButton>
        </div>
      </div>

      <div className="bg-slate-800 border border-slate-700 rounded-xl p-3 flex-1 overflow-hidden flex flex-col min-h-0">
        {isCalibration && hasVideo && (
          <div className="mb-3 flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between sm:gap-3 shrink-0">
            <div className="min-w-0 rounded-lg border border-slate-700 bg-slate-900/90 px-3 py-2">
              <div className="text-sm font-semibold text-slate-100">{calibrationHint}</div>
              <div className="mt-0.5 text-xs text-slate-400">
                点击画面会自动换算为原始视频像素坐标（已处理黑边/缩放）
              </div>
            </div>
            <div className="flex shrink-0 items-center gap-2 sm:justify-end">
              <Button variant="outline" className="py-1.5" onClick={props.onUndoCalibration} disabled={props.calibrationPoints.length === 0}>
                <Undo2 className="h-4 w-4" />
                撤销
              </Button>
              <Button variant="outline" className="py-1.5" onClick={props.onResetCalibration} disabled={props.calibrationPoints.length === 0}>
                <RotateCcw className="h-4 w-4" />
                重置
              </Button>
            </div>
          </div>
        )}

        <div className="aspect-video min-h-0 w-full bg-black rounded-xl relative overflow-hidden shrink-0">
          {!hasVideo && (
            <div className="absolute inset-0 p-4">
              <div className="h-full w-full bg-black/70 border border-slate-800 rounded-xl overflow-hidden">
                <div className="px-3 py-2 border-b border-slate-800 flex items-center justify-between">
                  <div className="text-xs text-slate-400 font-mono">Terminal · shuttlevision@local</div>
                  <div className="text-xs text-slate-500 font-mono">no-video</div>
                </div>
                <div className="p-3 font-mono text-xs text-slate-100 space-y-1">
                  <div className="text-slate-400">[00:00:00] Ready.</div>
                  <div className="text-emerald-400">[Hint] 请先导入羽毛球比赛视频。</div>
                  <div className="text-slate-400">[Hint] 建议 1080P 固定机位，完整覆盖整个场地</div>
                  <div className="text-slate-500">[UI] 上传后进入 6 点标定，点满即可开始分析…</div>
                </div>
              </div>
            </div>
          )}

          {props.videoObjectUrl && (
            <video
              ref={videoRef}
              className="absolute inset-0 h-full w-full object-contain opacity-90"
              src={props.videoObjectUrl}
              muted
              playsInline
              onLoadedMetadata={() => {
                const v = videoRef.current
                if (v) {
                  const d = Number.isFinite(v.duration) ? v.duration : 0
                  setVideoDuration(d)
                  setVideoTime(v.currentTime)
                  props.onVideoDurationChange?.(d)
                  props.onVideoTimeChange?.(v.currentTime)
                  setVideoPlaying(!v.paused)
                }
              }}
              onTimeUpdate={() => {
                const v = videoRef.current
                if (v) {
                  setVideoTime(v.currentTime)
                  props.onVideoTimeChange?.(v.currentTime)
                }
              }}
              onPlay={() => setVideoPlaying(true)}
              onPause={() => setVideoPlaying(false)}
              onEnded={() => setVideoPlaying(false)}
            />
          )}

          {isCalibration && hasVideo ? (
            <div className="absolute inset-0 z-10 cursor-crosshair" onClick={handleCanvasClick} role="presentation" />
          ) : null}

          {/* 分析成功后切换为结果 MP4，标定点的 display 坐标不再对齐画面，且易与「黑屏」混淆，故隐藏 */}
          {props.jobStatus !== 'succeeded'
            ? props.calibrationPoints.map((p, idx) => (
                <div
                  key={idx}
                  className="pointer-events-none absolute z-20 h-[7px] w-[7px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-emerald-400 shadow-[0_0_0_1.5px_rgba(255,255,255,0.95),0_0_4px_rgba(0,0,0,0.85)]"
                  style={{ left: `${p.displayX}px`, top: `${p.displayY}px` }}
                  title={`标定点 ${idx + 1} / ${CALIBRATION_PROMPTS[idx] ?? ''}`}
                />
              ))
            : null}

        </div>

        {hasVideo ? (
          <div className="mt-3 flex shrink-0 flex-col gap-2">
            <div className="flex items-center gap-3">
              <button
                type="button"
                className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-slate-600 bg-slate-700/80 text-slate-100 hover:bg-slate-600/90"
                onClick={() => void togglePlayPause()}
                aria-label={videoPlaying ? '暂停' : '播放'}
              >
                {videoPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4 pl-0.5" />}
              </button>
              <input
                type="range"
                className="min-w-0 flex-1 accent-emerald-500"
                min={0}
                max={videoDuration > 0 ? videoDuration : 0}
                step={0.01}
                value={Math.min(videoTime, videoDuration || 0)}
                onChange={(e) => seekTo(Number(e.target.value))}
                aria-label="视频进度"
              />
              <div className="shrink-0 font-mono text-xs text-slate-400 tabular-nums">
                {formatClock(videoTime)} / {formatClock(videoDuration)}
              </div>
            </div>
          </div>
        ) : null}

        <div className="mt-4">
          <div className="w-full h-2 bg-slate-700 rounded-full overflow-hidden">
            <div className="h-2 bg-emerald-500" style={{ width: `${Math.round(props.progress * 100)}%` }} />
          </div>
          <div className="mt-2 flex items-center justify-between text-xs text-slate-400">
            <div>分析进度</div>
            <div>{Math.round(props.progress * 100)}%</div>
          </div>
        </div>

        <div className="mt-3">
          <button
            className="w-full flex items-center justify-between text-sm text-slate-200 hover:text-slate-100 transition-colors"
            onClick={() => setIsLogsOpen((v) => !v)}
          >
            <span className="font-semibold">终端输出 (Terminal Log)</span>
            {isLogsOpen ? <ChevronDown className="h-4 w-4" /> : <ChevronUp className="h-4 w-4" />}
          </button>
          {isLogsOpen && (
            <div className="mt-2 h-32 bg-black font-mono text-xs p-2 rounded-lg overflow-y-auto border border-slate-800 text-slate-100">
              {props.logs.length === 0 ? (
                <div className="text-slate-500">尚无日志输出。</div>
              ) : (
                props.logs.slice(-200).map((l, i) => (
                  <div key={i} className="text-slate-300">
                    {l}
                  </div>
                ))
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

