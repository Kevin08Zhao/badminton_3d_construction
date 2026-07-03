import { CheckCircle2, XCircle } from 'lucide-react'
import { Button } from './ui'

type PerfMode = 'fast' | 'standard' | 'precise'

export function SettingsModal(props: {
  open: boolean
  onClose: () => void
  defaultMode: PerfMode
  batchSize: number
  weights: { label: string; path: string; ok: boolean }[]
  onChangeMode: (mode: PerfMode) => void
}) {
  if (!props.open) return null

  return (
    <div className="fixed inset-0 z-50">
      <div className="absolute inset-0 bg-black/60" onClick={props.onClose} />
      <div className="absolute inset-0 flex items-center justify-center p-4">
        <div className="w-full max-w-xl bg-slate-800 border border-slate-700 rounded-2xl shadow-xl overflow-hidden">
          <div className="p-4 flex items-center justify-between border-b border-slate-700">
            <div>
              <div className="text-sm font-semibold text-slate-100">高级设置</div>
              <div className="text-xs text-slate-400">
                性能模式会影响 batch_size 与处理耗时
              </div>
            </div>
            <Button variant="ghost" onClick={props.onClose}>
              关闭
            </Button>
          </div>

          <div className="p-4 space-y-4">
            <div className="bg-slate-900/40 border border-slate-700/60 rounded-xl p-4">
              <div className="text-sm font-semibold text-slate-100">硬件性能模式</div>
              <div className="mt-2 grid grid-cols-3 gap-2">
                <button
                  type="button"
                  onClick={() => props.onChangeMode('fast')}
                  className={[
                    'text-left border rounded-lg p-3 transition-colors',
                    props.defaultMode === 'fast'
                      ? 'border-emerald-500/40 bg-emerald-500/10'
                      : 'border-slate-700 bg-slate-800/60 hover:bg-slate-700/50',
                  ].join(' ')}
                >
                  <div className="text-sm text-slate-100 font-semibold">快</div>
                  <div className="text-xs text-slate-400 mt-1">更高吞吐，较低精度</div>
                </button>
                <button
                  type="button"
                  onClick={() => props.onChangeMode('standard')}
                  className={[
                    'text-left border rounded-lg p-3 transition-colors',
                    props.defaultMode === 'standard'
                      ? 'border-emerald-500/40 bg-emerald-500/10'
                      : 'border-slate-700 bg-slate-800/60 hover:bg-slate-700/50',
                  ].join(' ')}
                >
                  <div className="text-sm text-slate-100 font-semibold">标准</div>
                  <div className="text-xs text-slate-400 mt-1">默认推荐</div>
                </button>
                <button
                  type="button"
                  onClick={() => props.onChangeMode('precise')}
                  className={[
                    'text-left border rounded-lg p-3 transition-colors',
                    props.defaultMode === 'precise'
                      ? 'border-emerald-500/40 bg-emerald-500/10'
                      : 'border-slate-700 bg-slate-800/60 hover:bg-slate-700/50',
                  ].join(' ')}
                >
                  <div className="text-sm text-slate-100 font-semibold">精准</div>
                  <div className="text-xs text-slate-400 mt-1">更稳健，耗时更高</div>
                </button>
              </div>
              <div className="mt-3 text-xs text-slate-400">
                当前：<span className="text-slate-200 font-semibold">{props.defaultMode}</span> ·
                batch_size = <span className="font-mono text-slate-200">{props.batchSize}</span>
              </div>
            </div>

            <div className="bg-slate-900/40 border border-slate-700/60 rounded-xl p-4">
              <div className="text-sm font-semibold text-slate-100">模型权重路径检查</div>
              <div className="mt-3 space-y-2">
                {props.weights.map((w) => (
                  <div
                    key={w.label}
                    className="flex items-start justify-between gap-3 bg-slate-800/60 border border-slate-700 rounded-lg p-3"
                  >
                    <div className="min-w-0">
                      <div className="text-sm text-slate-200 font-semibold">{w.label}</div>
                      <div className="mt-1 text-xs text-slate-400 font-mono truncate">
                        {w.path}
                      </div>
                    </div>
                    <div className="pt-0.5">
                      {w.ok ? (
                        <div className="inline-flex items-center gap-1 text-emerald-400 text-xs font-semibold">
                          <CheckCircle2 className="h-4 w-4" />
                          OK
                        </div>
                      ) : (
                        <div className="inline-flex items-center gap-1 text-rose-400 text-xs font-semibold">
                          <XCircle className="h-4 w-4" />
                          Missing
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
              <div className="mt-3 text-xs text-slate-400">
                后续会接入真实文件检查与后端配置同步。
              </div>
            </div>
          </div>

          <div className="p-4 border-t border-slate-700 flex items-center justify-end gap-2">
            <Button variant="outline" onClick={props.onClose}>
              取消
            </Button>
            <Button variant="primary" onClick={props.onClose}>
              保存
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}

