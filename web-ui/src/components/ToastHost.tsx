import { AlertTriangle, CheckCircle2, Info } from 'lucide-react'

type Toast = {
  kind: 'warning' | 'success' | 'info'
  title: string
  message: string
}

function tone(kind: Toast['kind']) {
  if (kind === 'success')
    return {
      wrap: 'border-emerald-500/40 bg-emerald-500/10',
      title: 'text-emerald-200',
      icon: <CheckCircle2 className="h-4 w-4 text-emerald-400" />,
    }
  if (kind === 'warning')
    return {
      wrap: 'border-amber-500/40 bg-amber-500/10',
      title: 'text-amber-200',
      icon: <AlertTriangle className="h-4 w-4 text-amber-400" />,
    }
  return {
    wrap: 'border-sky-500/40 bg-sky-500/10',
    title: 'text-sky-200',
    icon: <Info className="h-4 w-4 text-sky-400" />,
  }
}

export function ToastHost(props: { toasts: Toast[] }) {
  if (props.toasts.length === 0) return null
  return (
    <div className="fixed top-4 right-4 z-[60] w-[360px] space-y-2">
      {props.toasts.map((t, i) => {
        const tt = tone(t.kind)
        return (
          <div
            key={`${i}-${t.kind}-${t.title}`}
            className={[
              'border rounded-xl p-3 backdrop-blur shadow-lg',
              'bg-slate-900/70 border-slate-700',
              tt.wrap,
            ].join(' ')}
          >
            <div className="flex items-start gap-2">
              <div className="pt-0.5">{tt.icon}</div>
              <div className="min-w-0">
                <div className={['text-sm font-semibold', tt.title].join(' ')}>{t.title}</div>
                <div className="text-xs text-slate-200/80 mt-0.5">{t.message}</div>
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}

