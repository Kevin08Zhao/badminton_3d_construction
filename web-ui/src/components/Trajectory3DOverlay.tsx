import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import {
  drawShuttleTrajectory3D,
  maxTrajectoryFrame,
  parseReconstructedTrajectoryCsv,
  videoTimeToFrameIndex,
  type ShotTrajectorySegment,
} from '../lib/shuttleTrajectory3d'

type Trajectory3DOverlayProps = {
  csvUrl: string | null
  /** 当前播放时间（秒） */
  currentTimeSec: number
  /** 视频时长（秒），用于在无 meta fps 时从 CSV 最大帧估算帧率 */
  videoDurationSec: number
  /** 后端 meta.video_fps；缺省时用 maxFrame/duration 或 30 */
  fpsHint?: number | null
  /** 是否显示（分析完成且有 CSV） */
  enabled: boolean
  /** true: 视频上浮层定位；false: 作为普通面板内组件渲染 */
  floating?: boolean
}

/**
 * 叠在视频左下角的小画布：按播放进度实时绘制 3D 世界系中的球路（多击球分段着色）。
 */
export function Trajectory3DOverlay(props: Trajectory3DOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [segments, setSegments] = useState<ShotTrajectorySegment[]>([])
  const [loadError, setLoadError] = useState<string | null>(null)

  useEffect(() => {
    setSegments([])
    setLoadError(null)
    if (!props.enabled || !props.csvUrl) return

    const url = props.csvUrl
    const ac = new AbortController()
    ;(async () => {
      try {
        const res = await fetch(url, { signal: ac.signal })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const text = await res.text()
        const parsed = parseReconstructedTrajectoryCsv(text)
        setSegments(parsed)
      } catch (e) {
        if (ac.signal.aborted) return
        setLoadError(e instanceof Error ? e.message : String(e))
        setSegments([])
      }
    })()

    return () => ac.abort()
  }, [props.enabled, props.csvUrl])

  const maxFrame = useMemo(() => maxTrajectoryFrame(segments), [segments])

  const fps = useMemo(() => {
    const hint = props.fpsHint
    if (typeof hint === 'number' && Number.isFinite(hint) && hint > 1) return hint
    const d = props.videoDurationSec
    if (maxFrame > 0 && Number.isFinite(d) && d > 0.05) {
      const est = maxFrame / d
      if (est > 1 && est < 480) return est
    }
    return 30
  }, [props.fpsHint, props.videoDurationSec, maxFrame])

  const currentFrame = useMemo(() => {
    const f = videoTimeToFrameIndex(props.currentTimeSec, fps)
    return maxFrame > 0 ? Math.min(f, maxFrame) : f
  }, [props.currentTimeSec, fps, maxFrame])

  useLayoutEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !props.enabled) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const w = canvas.width
    const h = canvas.height
    if (loadError) {
      ctx.fillStyle = 'rgba(15, 23, 42, 0.92)'
      ctx.fillRect(0, 0, w, h)
      ctx.fillStyle = '#f87171'
      ctx.font = '10px ui-sans-serif, system-ui, sans-serif'
      ctx.fillText(loadError.slice(0, 80), 8, h / 2)
      return
    }

    drawShuttleTrajectory3D(ctx, w, h, segments, {
      currentFrame,
      showFutureGhost: true,
    })
  }, [props.enabled, segments, currentFrame, loadError])

  if (!props.enabled || !props.csvUrl) return null

  const floating = props.floating ?? true
  const rootClass = floating
    ? 'pointer-events-none absolute left-2 bottom-2 z-[15] flex flex-col gap-1'
    : 'flex flex-col gap-1 w-full h-full'

  return (
    <div className={rootClass}>
      <canvas
        ref={canvasRef}
        width={240}
        height={200}
        className={[
          'rounded-lg border border-slate-600/90 shadow-lg shadow-black/40',
          floating ? '' : 'w-full h-full object-contain bg-slate-950/70',
        ].join(' ')}
        aria-label="3D 球路轨迹"
      />
      <div className="text-[10px] text-slate-300/90 font-mono px-0.5 drop-shadow-md">3D 轨迹 · 同步播放</div>
    </div>
  )
}
