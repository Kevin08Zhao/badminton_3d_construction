import type { ReactNode } from 'react'

export function Card(props: {
  title?: string
  right?: ReactNode
  children: ReactNode
  className?: string
}) {
  return (
    <section
      className={[
        'bg-slate-800 border border-slate-700 rounded-xl p-4 overflow-hidden min-w-0',
        props.className,
      ]
        .filter(Boolean)
        .join(' ')}
    >
      {(props.title || props.right) && (
        <header className="flex items-center justify-between mb-3">
          {props.title ? (
            <h2 className="text-sm font-semibold tracking-wide text-slate-100">
              {props.title}
            </h2>
          ) : (
            <div />
          )}
          {props.right}
        </header>
      )}
      {props.children}
    </section>
  )
}

export function Badge(props: { children: ReactNode; tone?: 'emerald' | 'slate' }) {
  const tone =
    props.tone === 'emerald'
      ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30'
      : 'bg-slate-700/40 text-slate-300 border-slate-600/40'
  return (
    <span
      className={[
        'inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs border',
        tone,
      ].join(' ')}
    >
      {props.children}
    </span>
  )
}

export function Button(
  props: React.ButtonHTMLAttributes<HTMLButtonElement> & {
    variant?: 'primary' | 'ghost' | 'outline'
    fullWidth?: boolean
  },
) {
  const { variant: v, fullWidth, className: cn, ...rest } = props
  const base =
    'inline-flex items-center justify-center gap-2 rounded-lg px-3 py-2 text-sm font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed'
  const variant =
    v === 'ghost'
      ? 'bg-transparent hover:bg-slate-700/60 text-slate-100'
      : v === 'outline'
        ? 'bg-transparent border border-slate-600 hover:bg-slate-700 text-slate-100'
        : 'bg-emerald-600 hover:bg-emerald-500 text-white'
  const width = fullWidth ? 'w-full' : ''
  const className = [base, variant, width, cn].filter(Boolean).join(' ')
  return <button {...rest} className={className} />
}

export function IconButton(
  props: React.ButtonHTMLAttributes<HTMLButtonElement> & { label: string },
) {
  return (
    <button
      {...props}
      aria-label={props.label}
      className={[
        'inline-flex items-center justify-center rounded-lg p-2',
        'bg-slate-800 border border-slate-700 hover:bg-slate-700/60 transition-colors',
        props.className,
      ]
        .filter(Boolean)
        .join(' ')}
    />
  )
}

