import { useMemo, useState } from 'react'
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { ChevronDown, ChevronUp, Download } from 'lucide-react'
import { Button, Card } from './ui'

type ChartPoint = { frame: number; z: number; speed: number }

export type ChartViewId =
  | 'height_speed'
  | 'heatmap'
  | 'report_overview'

const CHART_OPTIONS: { id: ChartViewId; label: string }[] = [
  { id: 'height_speed', label: '击球高度 / 速度曲线' },
  { id: 'heatmap', label: '球员跑动分布热力图' },
  { id: 'report_overview', label: '综合六视图总图' },
]

type DownloadDef = { id: string; label: string; artifactKey: string }

const DOWNLOAD_OPTIONS: DownloadDef[] = [
  { id: 'mp4', label: '可视化视频 MP4', artifactKey: 'mp4' },
  { id: 'csv', label: '3D 轨迹 CSV', artifactKey: 'csv' },
  { id: 'heatmap', label: '球员跑动分布热力图', artifactKey: 'heatmap' },
  { id: 'report_png', label: '综合六视图总图 PNG', artifactKey: 'png' },
]

function ChartPlaceholder(props: { title: string; hint?: string }) {
  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center rounded-lg border border-dashed border-slate-600 bg-slate-900/40 p-4 text-center">
      <div className="text-sm font-semibold text-slate-300">{props.title}</div>
      <div className="mt-2 text-xs text-slate-500 max-w-sm">
        {props.hint ?? '后端生成对应图像后将自动启用下载；预览可接入静态图 URL 或 WebGL。'}
      </div>
    </div>
  )
}

function ArtifactPreview(props: {
  title: string
  url: string | null
  media: 'image' | 'video'
}) {
  if (!props.url) {
    return (
      <ChartPlaceholder
        title={props.title}
        hint="当前任务暂无该产物；分析完成后会自动显示预览。"
      />
    )
  }
  if (props.media === 'video') {
    return (
      <div className="absolute inset-0 bg-black">
        <video
          src={props.url}
          className="h-full w-full object-contain"
          controls
          playsInline
          preload="metadata"
        />
      </div>
    )
  }
  return (
    <div className="absolute inset-0 bg-slate-950">
      <img
        src={props.url}
        alt={props.title}
        className="h-full w-full object-contain"
        loading="lazy"
      />
    </div>
  )
}

function HeightSpeedChart(props: { data: ChartPoint[] }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={props.data} margin={{ left: 8, right: 10, top: 10, bottom: 0 }}>
        <CartesianGrid stroke="rgba(51,65,85,0.6)" strokeDasharray="4 4" />
        <XAxis
          dataKey="frame"
          tick={{ fill: '#94a3b8', fontSize: 11 }}
          axisLine={{ stroke: 'rgba(51,65,85,1)' }}
          tickLine={{ stroke: 'rgba(51,65,85,1)' }}
        />
        <YAxis
          yAxisId="left"
          tick={{ fill: '#94a3b8', fontSize: 11 }}
          axisLine={{ stroke: 'rgba(51,65,85,1)' }}
          tickLine={{ stroke: 'rgba(51,65,85,1)' }}
          tickFormatter={(v) => `${v.toFixed(1)}`}
        />
        <YAxis
          yAxisId="right"
          orientation="right"
          tick={{ fill: '#94a3b8', fontSize: 11 }}
          axisLine={{ stroke: 'rgba(51,65,85,1)' }}
          tickLine={{ stroke: 'rgba(51,65,85,1)' }}
          tickFormatter={(v) => `${Math.round(v)}`}
        />
        <Tooltip
          contentStyle={{
            background: 'rgba(15, 23, 42, 0.95)',
            border: '1px solid rgba(51, 65, 85, 1)',
            borderRadius: 8,
            color: '#e2e8f0',
            fontSize: 12,
          }}
          labelStyle={{ color: '#e2e8f0' }}
        />
        <Legend wrapperStyle={{ color: '#cbd5e1', fontSize: 11 }} />
        <Line
          yAxisId="left"
          type="monotone"
          dataKey="z"
          name="高度 Z (m)"
          stroke="#10b981"
          strokeWidth={2}
          dot={false}
        />
        <Line
          yAxisId="right"
          type="monotone"
          dataKey="speed"
          name="速度 (m/s)"
          stroke="#60a5fa"
          strokeWidth={2}
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}

export function AnalysisExportPanel(props: {
  data: ChartPoint[]
  hasData: boolean
  artifacts: Record<string, string> | null
  disabled: boolean
  className?: string
}) {
  const [view, setView] = useState<ChartViewId>('height_speed')
  const [selected, setSelected] = useState<Set<string>>(() => new Set())
  const [filesPanelOpen, setFilesPanelOpen] = useState(false)

  const artifactMap = props.artifacts ?? {}
  const viewTitle = CHART_OPTIONS.find((o) => o.id === view)?.label ?? '图表预览'
  const currentArtifactUrl =
    view === 'height_speed'
      ? null
      : view === 'heatmap'
        ? artifactMap.heatmap ?? null
        : artifactMap.png ?? null

  const toggle = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const downloadableIds = useMemo(() => {
    return new Set(
      DOWNLOAD_OPTIONS.filter((o) => Boolean(artifactMap[o.artifactKey])).map((o) => o.id),
    )
  }, [artifactMap])

  function downloadSelected() {
    const urls: string[] = []
    for (const id of selected) {
      const def = DOWNLOAD_OPTIONS.find((d) => d.id === id)
      if (!def) continue
      const url = artifactMap[def.artifactKey]
      if (url) urls.push(url)
    }
    urls.forEach((u) => window.open(u, '_blank'))
  }

  const canDownload = !props.disabled && selected.size > 0

  return (
    <Card
      title="图表与导出"
      className={[
        'flex flex-col min-h-0 flex-1 p-3 [&_header]:mb-2',
        props.className,
      ]
        .filter(Boolean)
        .join(' ')}
    >
      <div className="flex items-center gap-2 shrink-0">
        <label htmlFor="chart-view" className="text-[11px] text-slate-400 whitespace-nowrap">
          图表类型
        </label>
        <select
          id="chart-view"
          value={view}
          onChange={(e) => setView(e.target.value as ChartViewId)}
          className="flex-1 min-w-0 rounded-lg border border-slate-600 bg-slate-900 px-2 py-1.5 text-xs text-slate-100"
        >
          {CHART_OPTIONS.map((o) => (
            <option key={o.id} value={o.id}>
              {o.label}
            </option>
          ))}
        </select>
      </div>

      <div className="mt-2 flex-1 min-h-0 min-w-0 relative rounded-lg border border-slate-700/80 bg-slate-950/50 overflow-hidden">
        {!props.hasData || props.data.length === 0 ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <div className="text-3xl font-semibold text-slate-600">--</div>
            <div className="mt-2 text-xs text-slate-500">分析完成后可查看图表</div>
          </div>
        ) : view === 'height_speed' ? (
          <HeightSpeedChart data={props.data} />
        ) : (
          <ArtifactPreview
            title={viewTitle}
            url={currentArtifactUrl}
            media="image"
          />
        )}
      </div>

      <div className="mt-2 pt-2 border-t border-slate-700 shrink-0 flex flex-col">
        {/* 上拉展开：列表在按钮行上方展开 */}
        <div
          className={[
            'overflow-hidden transition-[max-height] duration-200 ease-out border border-b-0 border-slate-700 rounded-t-lg bg-slate-900/60',
            filesPanelOpen ? 'max-h-52' : 'max-h-0 border-0',
          ].join(' ')}
        >
          <div className="px-3 py-2 max-h-48 overflow-y-auto">
            <div className="text-[11px] font-semibold text-slate-500 mb-2">勾选需要下载的文件（可多选）</div>
            <div className="space-y-2">
              {DOWNLOAD_OPTIONS.map((o) => {
                const ok = downloadableIds.has(o.id)
                return (
                  <label
                    key={o.id}
                    className={[
                      'flex items-center gap-2 text-xs cursor-pointer select-none',
                      ok ? 'text-slate-200' : 'text-slate-500 cursor-not-allowed',
                    ].join(' ')}
                  >
                    <input
                      type="checkbox"
                      disabled={!ok || props.disabled}
                      checked={selected.has(o.id)}
                      onChange={() => ok && toggle(o.id)}
                      className="rounded border-slate-600 shrink-0"
                    />
                    <span className="leading-tight">{o.label}</span>
                    {!ok && (
                      <span className="text-[10px] text-slate-600 ml-auto shrink-0">暂无</span>
                    )}
                  </label>
                )
              })}
            </div>
          </div>
        </div>

        <div
          className={[
            'flex items-center justify-between gap-3 px-1 py-2 rounded-b-lg',
            filesPanelOpen ? 'border border-slate-700 border-t-0 bg-slate-900/40' : '',
          ].join(' ')}
        >
          <Button variant="primary" disabled={!canDownload} onClick={downloadSelected} className="shrink-0">
            <Download className="h-4 w-4" />
            下载所选
            {selected.size > 0 ? ` (${selected.size})` : ''}
          </Button>
          <button
            type="button"
            disabled={props.disabled}
            onClick={() => setFilesPanelOpen((v) => !v)}
            className={[
              'inline-flex items-center gap-2 text-xs font-medium px-3 py-2 rounded-lg border transition-colors shrink-0',
              props.disabled
                ? 'border-slate-800 text-slate-600 cursor-not-allowed'
                : 'border-slate-600 text-slate-300 hover:bg-slate-700/60 hover:text-slate-100',
            ].join(' ')}
          >
            {filesPanelOpen ? (
              <>
                收起文件列表
                <ChevronDown className="h-4 w-4" />
              </>
            ) : (
              <>
                展开可下载文件
                <ChevronUp className="h-4 w-4" />
              </>
            )}
          </button>
        </div>
        <p className="mt-1 text-[10px] text-slate-500 leading-snug">
          先展开右侧勾选，再点左侧下载。后端就绪后对应项可选。
        </p>
      </div>
    </Card>
  )
}
