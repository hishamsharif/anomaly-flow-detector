import { useEffect, useRef, useState } from 'react'
import {
  batchPredict,
  getPredictJobResults,
  getPredictJobStatus,
  submitVerdict,
  type AlertResponse,
  type BatchPredictResponse,
  type RejectedRow,
} from '../api/client'

// ── Badge helpers ─────────────────────────────────────────────────────────────

const CONSENSUS_CLS: Record<string, string> = {
  SAFE: 'bg-green-900/60 text-green-300',
  KNOWN_ATTACK: 'bg-orange-900/60 text-orange-300',
  NOVEL_ATTACK: 'bg-red-900/60 text-red-400',
}

const SEVERITY_CLS: Record<string, string> = {
  low: 'bg-gray-800 text-gray-400',
  medium: 'bg-yellow-900/60 text-yellow-300',
  high: 'bg-orange-900/60 text-orange-300',
  critical: 'bg-red-900/60 text-red-400',
}

const VERDICT_CLS: Record<string, string> = {
  confirmed: 'bg-green-900/60 text-green-300',
  override: 'bg-blue-900/60 text-blue-300',
  uncertain: 'bg-gray-800 text-gray-400',
  escalate: 'bg-red-900/60 text-red-400',
}

function Badge({ text, cls }: { text: string; cls: string }) {
  return <span className={`text-xs px-2 py-0.5 rounded font-medium ${cls}`}>{text}</span>
}

// ── Upload area ───────────────────────────────────────────────────────────────

function UploadArea({ onFile }: { onFile: (f: File) => void }) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  const ACCEPTED = ['.csv', '.pcap', '.cap']
  const accept = (f: File) => {
    if (ACCEPTED.some(ext => f.name.toLowerCase().endsWith(ext))) onFile(f)
  }

  return (
    <div
      onClick={() => inputRef.current?.click()}
      onDragOver={e => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={e => { e.preventDefault(); setDragging(false); const f = e.dataTransfer.files[0]; if (f) accept(f) }}
      className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
        ${dragging ? 'border-blue-500 bg-blue-900/10' : 'border-gray-700 hover:border-gray-500'}`}
    >
      <p className="text-sm text-gray-400">Drop a <span className="text-gray-200">.csv</span> or <span className="text-gray-200">.pcap</span> file here, or <span className="text-blue-400 underline">click to browse</span></p>
      <p className="text-xs text-gray-600 mt-1">CSV must match the model's feature set · PCAP is parsed via nfstream</p>
      <input ref={inputRef} type="file" accept=".csv,.pcap,.cap" className="hidden" onChange={e => { const f = e.target.files?.[0]; if (f) accept(f); e.target.value = '' }} />
    </div>
  )
}

// ── Verdict buttons ───────────────────────────────────────────────────────────

function VerdictControls({
  alert,
  onUpdate,
}: {
  alert: AlertResponse
  onUpdate: (updated: AlertResponse) => void
}) {
  const [pending, setPending] = useState<string | null>(null)
  const [overrideMode, setOverrideMode] = useState(false)
  const [overrideLabel, setOverrideLabel] = useState('')
  const [err, setErr] = useState<string | null>(null)

  if (alert.analyst_verdict) {
    return <Badge text={alert.analyst_verdict} cls={VERDICT_CLS[alert.analyst_verdict] ?? 'bg-gray-800 text-gray-300'} />
  }

  const submit = async (verdict: string, label?: string) => {
    setPending(verdict)
    setErr(null)
    try {
      const updated = await submitVerdict(alert.id, verdict, label)
      onUpdate(updated)
    } catch (e: any) {
      setErr(e.message ?? 'Failed')
    } finally {
      setPending(null)
      setOverrideMode(false)
    }
  }

  if (overrideMode) {
    return (
      <div className="flex items-center gap-2">
        <input
          autoFocus
          value={overrideLabel}
          onChange={e => setOverrideLabel(e.target.value)}
          placeholder="label (e.g. DDoS)"
          className="bg-gray-800 border border-gray-600 text-white text-xs rounded px-2 py-1 w-28"
        />
        <button
          onClick={() => overrideLabel.trim() && submit('override', overrideLabel.trim())}
          disabled={!overrideLabel.trim() || !!pending}
          className="text-xs px-2 py-1 rounded bg-blue-700 text-white hover:bg-blue-600 disabled:opacity-40"
        >
          {pending === 'override' ? '…' : 'Save'}
        </button>
        <button onClick={() => setOverrideMode(false)} className="text-xs text-gray-400 hover:text-white">✕</button>
        {err && <span className="text-xs text-red-400">{err}</span>}
      </div>
    )
  }

  return (
    <div className="flex items-center gap-1 flex-wrap">
      {(['confirmed', 'uncertain', 'escalate'] as const).map(v => (
        <button
          key={v}
          onClick={() => submit(v)}
          disabled={!!pending}
          className="text-xs px-2 py-0.5 rounded border border-gray-700 text-gray-300 hover:bg-gray-700 disabled:opacity-40 capitalize"
        >
          {pending === v ? '…' : v}
        </button>
      ))}
      <button
        onClick={() => setOverrideMode(true)}
        disabled={!!pending}
        className="text-xs px-2 py-0.5 rounded border border-blue-800 text-blue-400 hover:bg-blue-900/40 disabled:opacity-40"
      >
        Override
      </button>
      {err && <span className="text-xs text-red-400">{err}</span>}
    </div>
  )
}

// ── Results table ─────────────────────────────────────────────────────────────

function ResultsTable({
  alerts,
  onUpdate,
}: {
  alerts: AlertResponse[]
  onUpdate: (updated: AlertResponse) => void
}) {
  const [expanded, setExpanded] = useState<number | null>(null)

  if (alerts.length === 0) return <p className="text-sm text-gray-500 py-4">No alerts in this batch.</p>

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-xs text-gray-500 border-b border-gray-800">
            <th className="text-left px-3 py-2">Flow</th>
            <th className="text-left px-3 py-2">Consensus</th>
            <th className="text-left px-3 py-2">Severity</th>
            <th className="text-left px-3 py-2">Conf.</th>
            <th className="text-left px-3 py-2">AE score</th>
            <th className="text-left px-3 py-2">Features</th>
            <th className="text-left px-3 py-2">Verdict</th>
          </tr>
        </thead>
        <tbody>
          {alerts.map(a => (
            <>
              <tr key={a.id} className="border-b border-gray-800 hover:bg-gray-800/30">
                <td className="px-3 py-2 text-gray-400 font-mono text-xs">{a.flow_id}</td>
                <td className="px-3 py-2">
                  <Badge text={a.consensus.replace('_', ' ')} cls={CONSENSUS_CLS[a.consensus] ?? 'bg-gray-800 text-gray-300'} />
                </td>
                <td className="px-3 py-2">
                  <Badge text={a.severity} cls={SEVERITY_CLS[a.severity] ?? 'bg-gray-800 text-gray-300'} />
                </td>
                <td className="px-3 py-2 text-gray-300">
                  {a.classifier_confidence != null ? (a.classifier_confidence * 100).toFixed(1) + '%' : '—'}
                </td>
                <td className="px-3 py-2 text-gray-300 font-mono text-xs">{a.ae_score.toFixed(4)}</td>
                <td className="px-3 py-2">
                  {a.top_features.length > 0 && (
                    <button
                      onClick={() => setExpanded(expanded === a.id ? null : a.id)}
                      className="text-xs text-blue-400 hover:underline"
                    >
                      {expanded === a.id ? 'hide' : `top ${a.top_features.length}`}
                    </button>
                  )}
                </td>
                <td className="px-3 py-2">
                  <VerdictControls alert={a} onUpdate={onUpdate} />
                </td>
              </tr>
              {expanded === a.id && (
                <tr key={`${a.id}-features`} className="border-b border-gray-800 bg-gray-900/60">
                  <td colSpan={7} className="px-3 py-2">
                    <div className="flex flex-wrap gap-2">
                      {a.top_features.map(f => (
                        <span key={f.name} className="text-xs bg-gray-800 rounded px-2 py-0.5 text-gray-300">
                          {f.name}: <span className="text-blue-300">{f.contribution.toFixed(4)}</span>
                        </span>
                      ))}
                    </div>
                  </td>
                </tr>
              )}
            </>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function InferencePage() {
  const [file, setFile] = useState<File | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [pollingPct, setPollingPct] = useState<number | null>(null)
  const [result, setResult] = useState<BatchPredictResponse | null>(null)
  const [alerts, setAlerts] = useState<AlertResponse[]>([])
  const [rejected, setRejected] = useState<RejectedRow[]>([])
  const [error, setError] = useState<string | null>(null)
  const [showRejected, setShowRejected] = useState(false)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopPolling = () => { if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null } }
  useEffect(() => () => stopPolling(), [])

  const handleSubmit = async () => {
    if (!file) return
    setError(null)
    setResult(null)
    setAlerts([])
    setRejected([])
    setPollingPct(null)
    setSubmitting(true)

    try {
      const res = await batchPredict(file)

      if (res.kind === 'sync') {
        setResult(res.data)
        setAlerts(res.data.alerts)
        setRejected(res.data.rejected_rows)
      } else {
        // Async job — poll until done
        setPollingPct(0)
        setSubmitting(false)
        const { job_id } = res
        pollRef.current = setInterval(async () => {
          try {
            const status = await getPredictJobStatus(job_id)
            setPollingPct(status.pct)
            if (status.done) {
              stopPolling()
              setPollingPct(null)
              const data = await getPredictJobResults(job_id)
              setResult(data)
              setAlerts(data.alerts)
              setRejected(data.rejected_rows)
            }
          } catch {
            stopPolling()
            setPollingPct(null)
            setError('Failed to poll job status')
          }
        }, 1500)
      }
    } catch (e: any) {
      setError(e.message ?? 'Inference failed')
    } finally {
      setSubmitting(false)
    }
  }

  const handleAlertUpdate = (updated: AlertResponse) => {
    setAlerts(prev => prev.map(a => (a.id === updated.id ? updated : a)))
  }

  return (
    <div className="max-w-6xl space-y-6">
      <h2 className="text-lg font-semibold text-white">Inference</h2>

      {/* Upload + submit */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-5 space-y-4">
        <UploadArea onFile={f => { setFile(f); setResult(null); setAlerts([]); setRejected([]) }} />

        {file && (
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-300">
              <span className="text-gray-500">Selected: </span>{file.name}
              <span className="text-gray-500 ml-2">({(file.size / 1024).toFixed(1)} KB)</span>
            </span>
            <button
              onClick={handleSubmit}
              disabled={submitting || pollingPct != null}
              className="px-4 py-2 rounded text-sm font-medium bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {submitting ? 'Submitting…' : 'Run Inference'}
            </button>
          </div>
        )}

        {/* Async progress bar */}
        {pollingPct != null && (
          <div>
            <div className="flex justify-between text-xs text-gray-400 mb-1">
              <span>Processing (async job)…</span>
              <span>{pollingPct}%</span>
            </div>
            <div className="w-full bg-gray-800 rounded-full h-2">
              <div className="bg-blue-500 h-2 rounded-full transition-all duration-300" style={{ width: `${pollingPct}%` }} />
            </div>
          </div>
        )}

        {error && (
          <p className="text-sm text-red-400 bg-red-900/20 border border-red-800 rounded px-3 py-2">{error}</p>
        )}
      </div>

      {/* Results */}
      {result && (
        <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
          {/* Summary bar */}
          <div className="px-5 py-3 border-b border-gray-800 flex flex-wrap gap-4 items-center">
            <h3 className="text-sm font-medium text-gray-300">Results</h3>
            <div className="flex gap-3 text-xs ml-auto">
              <span className="text-gray-400">Total: <span className="text-white">{result.summary.total}</span></span>
              <span className="text-green-300">Safe: {result.summary.safe}</span>
              <span className="text-orange-300">Known: {result.summary.known_attack}</span>
              <span className="text-red-400">Novel: {result.summary.novel_attack}</span>
              {result.summary.rejected > 0 && (
                <span className="text-gray-500">Rejected: {result.summary.rejected}</span>
              )}
            </div>
          </div>

          <div className="p-3">
            <ResultsTable alerts={alerts} onUpdate={handleAlertUpdate} />
          </div>

          {/* Rejected rows */}
          {rejected.length > 0 && (
            <div className="border-t border-gray-800 px-5 py-3">
              <button
                onClick={() => setShowRejected(v => !v)}
                className="text-xs text-gray-400 hover:text-white"
              >
                {showRejected ? '▾' : '▸'} {rejected.length} rejected row{rejected.length > 1 ? 's' : ''}
              </button>
              {showRejected && (
                <div className="mt-2 space-y-1">
                  {rejected.map((r, i) => (
                    <p key={i} className="text-xs text-gray-500 font-mono">Row {r.row}: {r.reason}</p>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
