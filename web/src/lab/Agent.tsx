import { Alert, Button, Chip } from '@heroui/react'
import { Bot, CircleStop, History, LoaderCircle, Play, RefreshCw, ScrollText, TerminalSquare } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { lab, type AiEvent, type AiRun, type Brief, type Harness, type HarnessId, type Version } from '../api'
import { HowTo, Picker, Section, Tip, fmt } from '../ui'
import type { LabState } from './useLab'

const AUTH = {
  ready: ['success', 'готов'], unknown: ['warning', 'вход не проверен'], login: ['danger', 'нужен вход'],
  missing: ['default', 'не установлен'], broken: ['danger', 'установка сломана'],
} as const
const FIELD = 'h-10 rounded-xl border border-border bg-surface px-3 text-sm focus-visible:outline-2 focus-visible:outline-focus'
const VSTATUS: Record<Version['status'], string> = {
  draft: 'черновик', evaluating: 'тестируется', candidate: 'прошла проверку', failed: 'не прошла проверку', promoted: 'была активной',
}
const FAMILY = { harsh0: 'Жёсткие миры', harsh50: 'Полужёсткие', stress: 'Стресс-миры' } as const

// выбор агента — удобство этого браузера: переживает переход между вкладками и перезагрузку
const PICK_KEY = 'cockpit.lab.agent'
const readPick = () => { try { return localStorage.getItem(PICK_KEY) ?? '' } catch { return '' } }
const savePick = (v: string) => { try { localStorage.setItem(PICK_KEY, v) } catch { /* приватный режим */ } }

const cap = (x: string) => x.charAt(0).toUpperCase() + x.slice(1)
const who = (h: HarnessId, model?: string | null) => (h === 'claude' ? `Claude${model ? ` ${cap(model)}` : ''}` : `Codex${model ? ` ${model}` : ''}`)
const mln = (x: number | null | undefined) => (x == null ? '—' : `${fmt(x / 1e6, 2)} млн`)
const delta = (a: number | null | undefined, b: number | null | undefined) => (a == null || b == null || !a ? null : (b - a) / Math.abs(a))
const ago = (iso: string) => {
  const m = Math.round((Date.now() - new Date(iso).getTime()) / 60000)
  return m < 1 ? 'только что' : m < 60 ? `${m} мин назад` : m < 1440 ? `${Math.round(m / 60)} ч назад`
    : new Date(iso).toLocaleDateString('ru', { day: 'numeric', month: 'short' })
}
const versionLabel = (v: Version) =>
  `${v.id} — ${v.current ? 'текущая, в agent.py' : VSTATUS[v.status]}${v.kind === 'ai' ? ' · правка AI' : v.kind === 'baseline' ? ' · исходная' : ''}`

// итог запуска словами: для списка и шапки карточки
function outcome(r: AiRun): { tone: 'accent' | 'success' | 'default' | 'danger' | 'warning'; text: string } {
  if (r.status === 'running') return { tone: 'accent', text: r.steps > 1 ? `Работает · попытка ${r.step} из ${r.steps}` : 'Работает' }
  if (r.status === 'cancelled') return { tone: 'default', text: 'Остановлен' }
  if (r.status === 'failed') return { tone: 'danger', text: 'Ошибка' }
  if (r.accepted.length) return { tone: 'success', text: `Найдено улучшение → ${r.best}` }
  return { tone: 'warning', text: r.versions.length ? 'Правки не прошли проверку' : 'Улучшить не удалось' }
}

export default function Agent({ s, open }: { s: LabState; open: (id: string) => void }) {
  const [hs, setHs] = useState<Harness[]>()
  const [agent, setAgent] = useState(readPick)
  const [task, setTask] = useState('')
  const [steps, setSteps] = useState(3)
  const [err, setErr] = useState<string>()
  const [runs, setRuns] = useState<AiRun[]>([])
  const [sel, setSel] = useState<string>()
  const loadHs = useCallback((refresh = false) => lab.harnesses(refresh).then((r) => setHs(r.harnesses), (e) => setErr(String(e))), [])
  const loadRuns = useCallback(() => lab.aiRuns().then(setRuns, (e) => setErr(String(e))), [])
  useEffect(() => { loadHs(); loadRuns() }, [loadHs, loadRuns])
  const active = runs.find((r) => r.status === 'running')
  // запуск идёт на сервере: пока он жив — опрашиваем список (статус, попытки)
  useEffect(() => {
    if (!active) return
    const t = setInterval(loadRuns, 2000)
    return () => clearInterval(t)
  }, [active, loadRuns])

  const ready = (hs ?? []).filter((h) => h.ready)
  // значение Picker — harness:model, пустая модель — по умолчанию CLI
  const options: [string, string][] = ready.flatMap((h) => [
    ...h.models.map((m) => [`${h.id}:${m}`, who(h.id, m)] as [string, string]),
    [`${h.id}:`, `${h.title} · модель по умолчанию`] as [string, string],
  ])
  const pick = options.some(([v]) => v === agent) ? agent : options[0]?.[0] ?? ''
  const [hid, model] = pick.split(':') as [HarnessId, string]
  const parent = s.selected
  const cur = runs.find((r) => r.id === sel) ?? runs[0]

  const start = () => {
    if (!parent) return
    setErr(undefined)
    lab.aiStart({ parent_id: parent.id, harness: hid, model, task: task.trim(), steps, budget_usd: null })
      .then((r) => { setSel(r.id); setTask(''); loadRuns() }, (e) => setErr(String(e)))
  }

  return (
    <>
      <HowTo>
        Опишите, что улучшить в стратегии, — <b>AI-агент</b> (Claude или Codex на этом компьютере) сам изменит код
        <code> agent.py</code>, проверит себя на тех же мирах, что и лаборатория, и сдаст правку, только если она проходит проверку.
        Каждая сданная правка — новая версия во «Версиях». При нескольких попытках агент учится на отказах: следующая
        попытка знает, почему не прошла предыдущая. Работает на сервере — вкладку можно закрыть.
      </HowTo>

      {hs && !ready.length && (
        <Section icon={TerminalSquare} title="Агент не готов" desc="Нужен установленный CLI с выполненным входом на машине сервера"
          action={<Button size="sm" variant="secondary" onPress={() => loadHs(true)}><RefreshCw className="size-3.5" aria-hidden />Проверить снова</Button>}>
          <HarnessList hs={hs} />
        </Section>
      )}

      <Section icon={Bot} title="Новая задача"
        desc={hs ? <span className="flex flex-wrap items-center gap-1.5">Доступны: {ready.length ? ready.map((h) => <Tip key={h.id} tip={h.version ?? ''}><Chip size="sm" variant="soft" color="success">{h.title}</Chip></Tip>) : 'нет'}
          <Tip tip={<HarnessList hs={hs} />}><button type="button" className="cursor-pointer text-xs text-muted underline" onClick={() => loadHs(true)}>обновить</button></Tip></span>
          : 'проверяю, какие агенты установлены…'}
        action={
          <Tip tip={active ? 'Агент уже работает — дождитесь или остановите его' : 'Агент начнёт работу на сервере'}>
            <Button variant="primary" isDisabled={!!active || !parent || !pick || task.trim().length < 3} onPress={start}>
              <Play className="size-4" aria-hidden />Запустить агента
            </Button>
          </Tip>
        }>
        {err && <p className="text-sm text-danger">{err}</p>}
        <textarea value={task} onChange={(e) => setTask(e.target.value)} rows={3} maxLength={4000} aria-label="Что нужно улучшить"
          placeholder="Что нужно улучшить? Например: «сократи лишние пилоты в маленьких ячейках — в худшем мире они съедают бюджет»"
          className="rounded-xl border border-border bg-surface px-3 py-2 text-sm focus-visible:outline-2 focus-visible:outline-focus" />
        {parent && parent.issues.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5 text-xs text-muted">
            Идеи из найденных проблем:
            {parent.issues.map((i) => (
              <Tip key={i.code} tip={i.evidence}>
                <Button size="sm" variant="tertiary" onPress={() => setTask(`${i.title}: ${i.evidence}. Исправь это.`)}>{i.title}</Button>
              </Tip>
            ))}
          </div>
        )}
        <div className="flex flex-wrap items-end gap-4">
          <label className="flex flex-col gap-1 text-xs font-medium text-muted">Что улучшаем
            <Picker label="Что улучшаем" className="w-80" value={parent?.id ?? ''} onChange={s.setSel}
              options={[...s.versions].reverse().map((v) => [v.id, versionLabel(v)])} />
          </label>
          {options.length > 0 && (
            <label className="flex flex-col gap-1 text-xs font-medium text-muted">Кто работает
              <Picker label="Кто работает" className="w-64" value={pick} onChange={(v) => { setAgent(v); savePick(v) }} options={options} />
            </label>
          )}
          <Tip tip="Сколько раз агент может попробовать. Правка прошла проверку — следующая попытка улучшает уже её; не прошла — агент узнаёт почему и пробует иначе">
            <label className="flex flex-col gap-1 text-xs font-medium text-muted">Попыток
              <input type="number" aria-label="Попыток" min={1} max={10} value={steps}
                onChange={(e) => setSteps(Math.min(10, Math.max(1, Number(e.target.value) || 1)))} className={`${FIELD} num w-24`} />
            </label>
          </Tip>
        </div>
      </Section>

      {runs.length > 0 && (
        <div className="grid gap-5 xl:grid-cols-[minmax(320px,1fr)_2fr]">
          <Section icon={History} title="История задач">
            <ol className="flex flex-col gap-1">
              {runs.map((r) => {
                const o = outcome(r)
                return (
                  <li key={r.id}>
                    <button type="button" onClick={() => setSel(r.id)} aria-current={r.id === cur?.id ? 'true' : undefined}
                      className={`flex w-full cursor-pointer flex-col gap-1 rounded-xl px-3 py-2.5 text-left transition-colors
                        focus-visible:outline-2 focus-visible:outline-focus ${r.id === cur?.id ? 'bg-accent-soft' : 'hover:bg-default'}`}>
                      <span className="line-clamp-2 text-sm font-medium">{r.task}</span>
                      <span className="flex flex-wrap items-center gap-2 text-xs">
                        <Chip size="sm" variant="soft" color={o.tone}>{r.status === 'running' && <LoaderCircle className="size-3 animate-spin" aria-hidden />}{o.text}</Chip>
                        <span className="text-muted">{who(r.harness, r.model)} · {ago(r.created_at)}</span>
                      </span>
                    </button>
                  </li>
                )
              })}
            </ol>
          </Section>
          {cur && <RunCard key={cur.id} brief={cur} open={open} onVersion={s.refresh} onStop={() => lab.aiCancel(cur.id).then(loadRuns, (e) => setErr(String(e)))} />}
        </div>
      )}
    </>
  )
}

function HarnessList({ hs }: { hs: Harness[] }) {
  return (
    <div className="flex flex-col gap-2 text-sm">
      {hs.map((h) => (
        <div key={h.id} className="flex flex-col gap-0.5">
          <span className="flex items-center gap-2"><b>{h.title}</b><Chip size="sm" variant="soft" color={AUTH[h.auth][0]}>{AUTH[h.auth][1]}</Chip></span>
          {!h.installed && <span className="text-xs text-muted">Установить: <code className="num">{h.install_cmd}</code></span>}
          {h.auth === 'login' && <span className="text-xs text-muted">Войти в терминале: <code className="num">{h.login_cmd}</code></span>}
        </div>
      ))}
    </div>
  )
}

// Задача целиком: итог → отчёт агента → сданные правки → подробный журнал. Лог дочитывается, пока агент работает
function RunCard({ brief, open, onVersion, onStop }: { brief: AiRun; open: (id: string) => void; onVersion: () => void; onStop: () => void }) {
  const [run, setRun] = useState<AiRun>(brief)
  const [events, setEvents] = useState<AiEvent[]>([])
  const bump = useRef(onVersion)
  useEffect(() => { bump.current = onVersion })
  useEffect(() => {
    let live = true, last = 0, timer = 0
    const tick = () => lab.aiRun(brief.id, last).then((r) => {
      if (!live) return
      const evs = r.events ?? []
      if (evs.length) {
        last = evs.at(-1)!.i
        setEvents((xs) => [...xs, ...evs])
        if (evs.some((e) => e.kind === 'evaluating' || e.kind === 'evaluated')) bump.current()  // новые версии → список лаборатории
      }
      setRun(r)
      if (r.status === 'running') timer = window.setTimeout(tick, 1500)
    }, () => { if (live) timer = window.setTimeout(tick, 3000) })
    tick()
    return () => { live = false; clearTimeout(timer) }
  }, [brief.id])

  const o = outcome(run)
  const running = run.status === 'running'
  // старые запуски без summary — последний ответ агента из лога
  const lastText = [...events].reverse().find((e) => e.kind === 'text')
  const summary = run.summary ?? (lastText && 'text' in lastText ? lastText.text : undefined)
  // одна строка на правку: «проверяется» заменяется итогом проверки
  const attempts = events.filter((e) => e.kind === 'evaluated' || e.kind === 'rejected'
    || (e.kind === 'evaluating' && !events.some((x) => x.kind === 'evaluated' && x.version === e.version)))
  const cost = run.cost_usd ? ` · ≈ $${fmt(run.cost_usd, 2)} по ценам API` : ''

  return (
    <Section icon={Bot} title={run.task} desc={`${who(run.harness, run.model)} · улучшает ${run.parent_id} · ${ago(run.created_at)}${cost}`}
      action={running && <Button size="sm" variant="danger" onPress={onStop}><CircleStop className="size-3.5" aria-hidden />Остановить</Button>}>
      <Alert status={o.tone === 'accent' || o.tone === 'default' ? 'default' : o.tone}>
        <Alert.Content>
          <Alert.Title className="flex items-center gap-2">{running && <LoaderCircle className="size-4 animate-spin" aria-hidden />}{o.text}</Alert.Title>
          <Alert.Description>
            {running ? 'Агент читает код, правит и проверяет себя. Сданные правки появятся ниже.'
              : run.status === 'failed' ? run.error
              : run.status === 'cancelled' ? 'Остановлен вручную. Уже сданные правки сохранены.'
              : run.accepted.length ? <>Версия <b>{run.best}</b> прошла проверку лаборатории.{' '}
                  <Button size="sm" variant="secondary" onPress={() => open(run.best)}>Открыть версию</Button></>
              : run.versions.length ? 'Агент сдавал правки, но лаборатория их отклонила — причины ниже.'
              : 'Агент не нашёл правку, которая проходит проверку, и ничего не менял. Его выводы — ниже.'}
          </Alert.Description>
        </Alert.Content>
      </Alert>

      {!running && summary && (
        <div className="flex flex-col gap-1">
          <h4 className="text-sm font-semibold">Отчёт агента</h4>
          <div className="flex flex-col gap-1.5 rounded-xl bg-default/60 px-4 py-3 text-sm leading-relaxed"><Md text={summary} /></div>
        </div>
      )}

      {attempts.length > 0 && (
        <div className="flex flex-col gap-2">
          <h4 className="text-sm font-semibold">Сданные правки</h4>
          {attempts.map((e, i) => <Attempt key={e.i} n={i + 1} e={e} open={open} />)}
        </div>
      )}

      <details open={running} className="rounded-xl border border-border">
        <summary className="flex cursor-pointer items-center gap-2 px-4 py-2.5 text-sm font-medium">
          <ScrollText className="size-4" aria-hidden />Что делал агент <span className="text-xs font-normal text-muted">— {events.length} действий</span>
        </summary>
        <ol className="flex max-h-[60vh] flex-col gap-1.5 overflow-y-auto border-t border-border p-4 text-sm">
          {events.map((e) => <li key={e.i}><Line e={e} /></li>)}
        </ol>
      </details>
    </Section>
  )
}

function Attempt({ n, e, open }: { n: number; e: AiEvent; open: (id: string) => void }) {
  if (e.kind === 'rejected') {
    return <div className="rounded-xl border border-border px-4 py-3 text-sm"><b>Правка {n}</b> · <span className="text-warning">не собралась</span> — {e.text}</div>
  }
  if (e.kind === 'evaluating') {
    return (
      <div className="flex items-center gap-2 rounded-xl border border-border px-4 py-3 text-sm">
        <LoaderCircle className="size-4 animate-spin" aria-hidden /><b>Правка {n}</b> · версия {e.version} проходит проверку…
      </div>
    )
  }
  if (e.kind !== 'evaluated') return null
  return (
    <div className={`flex flex-col gap-2 rounded-xl border px-4 py-3 text-sm ${e.passed ? 'border-success/40' : 'border-border'}`}>
      <span className="flex flex-wrap items-center gap-2">
        <b>Правка {n}</b>
        <Chip size="sm" variant="soft" color={e.passed ? 'success' : 'danger'}>{e.passed ? 'прошла проверку' : 'не прошла'}</Chip>
        <span className="text-xs text-muted">версия {e.version}</span>
        <Button size="sm" variant="tertiary" className="ml-auto" onPress={() => open(e.version)}>Открыть</Button>
      </span>
      <Deltas m={e.metrics} base={e.base} />
      {!e.passed && e.reasons.length > 0 && <span className="text-xs text-muted">Почему: {e.reasons.join('; ')}</span>}
    </div>
  )
}

// результат против базы: медиана и худший мир по семействам, изменение в процентах
function Deltas({ m, base }: { m: Brief; base?: Brief }) {
  return (
    <div className="grid gap-2 sm:grid-cols-3">
      {(Object.keys(FAMILY) as (keyof Brief)[]).map((k) => (
        <div key={k} className="flex flex-col rounded-lg bg-default/60 px-3 py-2">
          <span className="text-xs text-muted">{FAMILY[k]}{k === 'harsh0' && ' · главное'}</span>
          <span className="num">{mln(m[k].median)} <Pct d={delta(base?.[k].median, m[k].median)} /></span>
          <span className="num text-xs text-muted">худший {mln(m[k].min)} <Pct d={delta(base?.[k].min, m[k].min)} /></span>
        </div>
      ))}
    </div>
  )
}

// ponytail: markdown отчёта агента — заголовки, списки, **жирный**, `код`; таблицы и ссылки остаются текстом.
// React-узлы, не innerHTML: текст агента — недоверенные данные
function Md({ text }: { text: string }) {
  const inline = (t: string) => t.split(/(\*\*[^*]+\*\*|`[^`]+`)/).map((x, i) =>
    x.startsWith('**') && x.endsWith('**') && x.length > 4 ? <b key={i}>{x.slice(2, -2)}</b>
      : x.startsWith('`') && x.endsWith('`') && x.length > 2 ? <code key={i} className="num rounded bg-default px-1 text-xs">{x.slice(1, -1)}</code> : x)
  return text.split('\n').filter((l) => l.trim()).map((l, i) => {
    const h = l.match(/^#{1,4}\s+(.*)/), li = l.match(/^\s*(?:[-*]|\d+[.)])\s+(.*)/)
    return h ? <p key={i} className="mt-1 font-semibold">{inline(h[1])}</p>
      : li ? <p key={i} className="flex gap-2 pl-2"><span className="text-muted">•</span><span>{inline(li[1])}</span></p>
      : <p key={i}>{inline(l)}</p>
  })
}

function Pct({ d }: { d: number | null }) {
  if (d == null) return null
  return <span className={d > 0 ? 'text-success' : d < 0 ? 'text-danger' : 'text-muted'}>{d > 0 ? '+' : ''}{fmt(100 * d, 2)}%</span>
}

function Line({ e }: { e: AiEvent }) {
  switch (e.kind) {
    case 'step': return <p className="mt-2 border-t border-border pt-2 font-semibold">{e.text.replace(/^Шаг/, 'Попытка').replace('· база', '· улучшает')}</p>
    case 'started': return <span className="text-xs text-muted">▶ агент начал работу над {e.parent}</span>
    case 'text': return <p className="whitespace-pre-wrap">{e.text}</p>
    case 'log': return <span className="num block text-xs whitespace-pre-wrap text-muted">{e.text}</span>
    case 'tool': return <span className="num block text-xs text-muted">— {e.tool} {e.detail}</span>
    case 'info': return <span className="text-sm text-muted">ℹ {e.text}</span>
    case 'rejected': return <span className="text-sm text-warning">✗ правка не собралась: {e.text}</span>
    case 'result': return <span className={`text-xs ${e.ok ? 'text-success' : 'text-danger'}`}>{e.ok ? '✓ агент закончил' : `✗ ${e.text}`}</span>
    case 'error': return <span className="text-sm text-danger">✗ {e.message}</span>
    case 'evaluating': return <span className="text-sm text-muted">⏳ {e.version}: лаборатория проверяет правку…</span>
    case 'evaluated': return <span className={`text-sm ${e.passed ? 'text-success' : 'text-danger'}`}>{e.passed ? '✓' : '✗'} {e.version}: {e.passed ? 'прошла проверку' : 'не прошла'}</span>
  }
}
