import { CheckCircle2, Circle, Gauge, Loader2 } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { Card } from './ui'
import { Trajectory3DOverlay } from './Trajectory3DOverlay'
import {
  maxTrajectoryFrame,
  parseReconstructedTrajectoryCsv,
  videoTimeToFrameIndex,
  type ShotTrajectorySegment,
} from '../lib/shuttleTrajectory3d'

type Step = { title: string; subtitle?: string }
type Props = {
  workflow: { currentStep: number; steps: Step[] }
  hasAnalysisData: boolean
  trajectoryCsvUrl?: string | null
  trajectoryFpsHint?: number | null
  currentTimeSec?: number
  videoDurationSec?: number
}

function StepIcon(props: { state: 'done' | 'current' | 'todo' }) {
  if (props.state === 'done') return <CheckCircle2 className="h-5 w-5 text-emerald-500" />
  if (props.state === 'current')
    return <Loader2 className="h-5 w-5 text-emerald-500 animate-spin" />
  return <Circle className="h-5 w-5 text-slate-600" />
}

/** 左侧面板中的 3D 球路轨迹（不覆盖视频区域） */
function Trajectory3DPreview(props: {
  active: boolean
  csvUrl?: string | null
  fpsHint?: number | null
  currentTimeSec?: number
  videoDurationSec?: number
}) {
  return (
    <div className="flex-1 min-h-[200px] min-w-0 flex flex-col">
      <div className="text-xs tracking-wide text-slate-400 font-semibold mb-2">3D 轨迹 (球场坐标系)</div>
      <div className="flex-1 min-h-[180px] rounded-xl border border-slate-700 bg-gradient-to-b from-slate-950 to-black overflow-hidden relative">
        {!props.active || !props.csvUrl ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500 text-sm px-4 text-center">
            <span className="text-2xl font-mono text-slate-600 mb-2">--</span>
            分析完成后将在此绘制 3D 坐标系与球的实时轨迹
          </div>
        ) : (
          <div className="absolute inset-2">
            <Trajectory3DOverlay
              csvUrl={props.csvUrl ?? null}
              enabled={Boolean(props.active && props.csvUrl)}
              currentTimeSec={props.currentTimeSec ?? 0}
              videoDurationSec={props.videoDurationSec ?? 0}
              fpsHint={props.fpsHint ?? null}
              floating={false}
            />
          </div>
        )}
      </div>
    </div>
  )
}

export function WorkflowPanel(props: Props) {
  const { steps, currentStep } = props.workflow
  const d = props.hasAnalysisData
  const [segments, setSegments] = useState<ShotTrajectorySegment[]>([])

  useEffect(() => {
    if (!d || !props.trajectoryCsvUrl) {
      setSegments([])
      return
    }
    let cancelled = false
    void fetch(props.trajectoryCsvUrl)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.text()
      })
      .then((text) => {
        if (cancelled) return
        setSegments(parseReconstructedTrajectoryCsv(text))
      })
      .catch(() => {
        if (!cancelled) setSegments([])
      })
    return () => {
      cancelled = true
    }
  }, [d, props.trajectoryCsvUrl])

  const liveFps = useMemo(() => {
    const hint = props.trajectoryFpsHint
    if (typeof hint === 'number' && Number.isFinite(hint) && hint > 0) return hint
    const maxFrame = maxTrajectoryFrame(segments)
    const duration = props.videoDurationSec ?? 0
    if (maxFrame > 0 && duration > 0) return maxFrame / duration
    return 30
  }, [props.trajectoryFpsHint, props.videoDurationSec, segments])

  const liveMetrics = useMemo(() => {
    if (!d || segments.length === 0) return null
    const currentFrame = videoTimeToFrameIndex(props.currentTimeSec ?? 0, liveFps)

    type Candidate = { seg: ShotTrajectorySegment; idx: number; frame: number }
    let best: Candidate | null = null
    for (const seg of segments) {
      const pts = seg.points
      if (pts.length === 0) continue
      let lo = 0
      let hi = pts.length - 1
      let ans = -1
      while (lo <= hi) {
        const mid = (lo + hi) >> 1
        if (pts[mid].frame <= currentFrame) {
          ans = mid
          lo = mid + 1
        } else {
          hi = mid - 1
        }
      }
      if (ans < 0) continue
      const fr = pts[ans].frame
      if (!best || fr > best.frame) best = { seg, idx: ans, frame: fr }
    }
    if (!best) return null

    const cur = best.seg.points[best.idx]
    let speedMps: number | null = null
    if (best.idx > 0) {
      const prev = best.seg.points[best.idx - 1]
      const dt = (cur.frame - prev.frame) / liveFps
      if (dt > 1e-9) {
        const dx = cur.x - prev.x
        const dy = cur.y - prev.y
        const dz = cur.z - prev.z
        speedMps = Math.sqrt(dx * dx + dy * dy + dz * dz) / dt
      }
    } else if (best.seg.points.length > 1) {
      const next = best.seg.points[1]
      const dt = (next.frame - cur.frame) / liveFps
      if (dt > 1e-9) {
        const dx = next.x - cur.x
        const dy = next.y - cur.y
        const dz = next.z - cur.z
        speedMps = Math.sqrt(dx * dx + dy * dy + dz * dz) / dt
      }
    }

    return { x: cur.x, y: cur.y, z: cur.z, speedMps }
  }, [d, segments, props.currentTimeSec, liveFps])

  const fmtCoord = (v: number | null | undefined) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(2) : '--')
  const fmtSpeed = (v: number | null | undefined) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(1) : '--')

  return (
    <div className="h-full flex flex-col gap-4 overflow-hidden">
      <Card title="分析流水线 (Workflow)">
        <ol className="space-y-3">
          {steps.map((s, idx) => {
            const state: 'done' | 'current' | 'todo' =
              idx < currentStep ? 'done' : idx === currentStep ? 'current' : 'todo'
            const textTone =
              state === 'current'
                ? 'text-slate-100 font-semibold'
                : state === 'done'
                  ? 'text-slate-200'
                  : 'text-slate-500'
            return (
              <li key={s.title} className="flex items-start gap-3">
                <div className="pt-0.5">
                  <StepIcon state={state} />
                </div>
                <div className="min-w-0">
                  <div className={['text-sm leading-5 truncate', textTone].join(' ')}>
                    {idx + 1}. {s.title}
                  </div>
                  {s.subtitle && (
                    <div className="text-xs text-slate-500 truncate">{s.subtitle}</div>
                  )}
                </div>
              </li>
            )
          })}
        </ol>
      </Card>

      <Card title="实时坐标 (LIVE 3D)" className="flex-1 min-h-0 flex flex-col overflow-hidden">
        <div className="grid grid-cols-2 gap-3 shrink-0">
          <div className="bg-slate-900/50 border border-slate-700/60 rounded-xl p-3">
            <div className="text-xs text-slate-400">X</div>
            <div className="mt-1 text-3xl font-mono tracking-tight text-rose-400">
              {d ? fmtCoord(liveMetrics?.x) : '--'}
            </div>
          </div>
          <div className="bg-slate-900/50 border border-slate-700/60 rounded-xl p-3">
            <div className="text-xs text-slate-400">Y</div>
            <div className="mt-1 text-3xl font-mono tracking-tight text-emerald-400">
              {d ? fmtCoord(liveMetrics?.y) : '--'}
            </div>
          </div>
          <div className="bg-slate-900/50 border border-slate-700/60 rounded-xl p-3">
            <div className="text-xs text-slate-400">Z</div>
            <div className="mt-1 text-3xl font-mono tracking-tight text-sky-400">
              {d ? fmtCoord(liveMetrics?.z) : '--'}
            </div>
          </div>
          <div className="bg-slate-900/50 border border-slate-700/60 rounded-xl p-3">
            <div className="flex items-center gap-2 text-xs text-slate-400">
              <Gauge className="h-4 w-4 text-slate-300" />
              实时球速
            </div>
            <div className="mt-1 text-3xl font-mono tracking-tight text-slate-100">
              {d ? (
                <>
                  {fmtSpeed(liveMetrics?.speedMps)}
                  <span className="ml-2 text-xs text-slate-400 align-top">m/s</span>
                </>
              ) : (
                '--'
              )}
            </div>
          </div>
        </div>

        <div className="mt-4 flex-1 min-h-0 flex flex-col">
          <Trajectory3DPreview
            active={d}
            csvUrl={props.trajectoryCsvUrl}
            fpsHint={props.trajectoryFpsHint}
            currentTimeSec={props.currentTimeSec}
            videoDurationSec={props.videoDurationSec}
          />
        </div>
      </Card>
    </div>
  )
}
