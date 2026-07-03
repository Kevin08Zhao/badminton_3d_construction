import { Card } from './ui'

export type MetricsModel = {
  /** 检测击球次数 / Shots */
  detectionCount: number | null
  nearDistanceM: number | null
  farDistanceM: number | null
  nearAvgSpeedMps: number | null
  farAvgSpeedMps: number | null
} | null

type Props = {
  metrics: MetricsModel
}

function fmtM(v: number | null) {
  if (v == null) return '--'
  return `${v.toFixed(2)} m`
}

function fmtSpeed(v: number | null) {
  if (v == null) return '--'
  return `${v.toFixed(2)} m/s`
}

function MetricTile(props: {
  label: string
  value: string
  accent?: boolean
  /** 左侧跨行主指标：更大数值、撑满高度 */
  featured?: boolean
  className?: string
}) {
  const root = [
    'bg-slate-900/40 border border-slate-700/60 rounded-md',
    props.featured ? 'px-3 py-3 h-full flex flex-col justify-center min-h-0' : 'px-2 py-1.5',
    props.className ?? '',
  ]
    .filter(Boolean)
    .join(' ')
  const valueSize = props.featured ? 'text-2xl sm:text-3xl' : 'text-base'
  return (
    <div className={root}>
      <div className="text-[10px] text-slate-400 leading-snug line-clamp-2">{props.label}</div>
      <div
        className={[
          'mt-1 font-semibold tabular-nums tracking-tight',
          valueSize,
          props.featured ? '' : 'truncate',
          props.accent && props.value !== '--' ? 'text-emerald-400' : 'text-slate-100',
        ].join(' ')}
      >
        {props.value}
      </div>
    </div>
  )
}

export function KeyMetricsCard(props: Props) {
  const m = props.metrics
  const detStr = m?.detectionCount != null ? `${m.detectionCount}` : '--'

  return (
    <Card
      title="核心指标 (Key Metrics)"
      className="shrink-0 p-3 [&_header]:mb-2"
    >
      {/* flex：左侧按内容宽度，避免三等分列在左格内留白；与右侧 2×2 共用同一 gap */}
      <div className="flex gap-2 items-stretch min-h-[4.25rem]">
        <div className="shrink-0 w-[min(30%,8.75rem)] sm:w-[min(28%,9.5rem)] min-w-[6.75rem] self-stretch">
          <MetricTile label="检测击球" value={detStr} featured className="h-full w-full" />
        </div>
        <div className="min-w-0 flex-1 grid grid-cols-2 grid-rows-2 gap-2">
          <MetricTile label="远方路程" value={fmtM(m?.farDistanceM ?? null)} />
          <MetricTile label="近方路程" value={fmtM(m?.nearDistanceM ?? null)} />
          <MetricTile label="远方均速" value={fmtSpeed(m?.farAvgSpeedMps ?? null)} />
          <MetricTile label="近方均速" value={fmtSpeed(m?.nearAvgSpeedMps ?? null)} />
        </div>
      </div>
    </Card>
  )
}
