import { useEffect, useRef, useState } from 'react'
import {
  getActiveModel,
  getDashboardSummary,
  getModels,
  promoteModel,
  startTraining,
  subscribeToTraining,
  type DashboardSummary,
  type ModelVersionResponse,
} from '../api/client'

const STATUS_BADGE: Record<string, string> = {
  active: 'bg-green-900 text-green-300',
  training: 'bg-yellow-900 text-yellow-300',
  failed: 'bg-red-900 text-red-300',
  archived: 'bg-gray-800 text-gray-400',
}

export default function TrainingPage() {
  const [models, setModels] = useState<ModelVersionResponse[]>([])
  const [activeModel, setActiveModel] = useState<ModelVersionResponse | null>(null)
  const [summary, setSummary] = useState<DashboardSummary | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [promotingId, setPromotingId] = useState<number | null>(null)

  // HITL fine-tune state
  const [retraining, setRetraining] = useState(false)
  const [pct, setPct] = useState(0)
  const [logs, setLogs] = useState<string[]>([])
  const [retrainMsg, setRetrainMsg] = useState<{ text: string; ok: boolean } | null>(null)
  const esRef = useRef<EventSource | null>(null)
  const logsEndRef = useRef<HTMLDivElement>(null)

  const load = () => {
    getModels().then(data => setModels(data.items)).catch(() => {})
    getActiveModel().then(setActiveModel).catch(() => {})
    getDashboardSummary().then(setSummary).catch(() => {})
  }

  useEffect(() => {
    load()
    const id = setInterval(load, 30_000)
    return () => clearInterval(id)
  }, [])

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  const refreshModels = () =>
    getModels().then(data => setModels(data.items)).catch(() => {})

  const handlePromote = async (id: number) => {
    setPromotingId(id)
    setError(null)
    try {
      await promoteModel(id)
      await refreshModels()
    } catch (e: any) {
      setError(e.message ?? 'Failed to promote model')
    } finally {
      setPromotingId(null)
    }
  }

  const handleRetrain = async () => {
    setRetrainMsg(null)
    setError(null)
    setRetraining(true)
    setPct(0)
    setLogs([])
    try {
      const { run_id } = await startTraining({
        classifier_type: activeModel?.classifier_type ?? 'xgboost',
        detector_type:   activeModel?.detector_type   ?? 'mlp_ae',
        mode: 'ae_retrain',
      })
      const es = subscribeToTraining(run_id)
      esRef.current = es

      es.addEventListener('progress', e => setLogs(prev => [...prev, e.data]))
      es.addEventListener('step', e => setPct(parseInt(e.data, 10)))
      es.addEventListener('done', e => {
        setLogs(prev => [...prev, `✓ ${e.data}`])
        setPct(100)
        es.close()
        esRef.current = null
        setRetraining(false)
        setRetrainMsg({ text: 'AE fine-tune complete — new model version active.', ok: true })
        load()
      })
      es.addEventListener('error', e => {
        const msg = (e as MessageEvent).data ?? 'Fine-tune failed'
        setError(msg)
        setLogs(prev => [...prev, `✗ ${msg}`])
        es.close()
        esRef.current = null
        setRetraining(false)
      })
    } catch (e: any) {
      setError(e.message ?? 'Failed to start fine-tune')
      setRetraining(false)
    }
  }

  const verdictEligible  = summary?.verdict_eligible ?? 0
  const verdictThreshold = summary?.verdict_threshold ?? 50
  const retrainUnlocked  = summary?.retrain_unlocked ?? false
  const progressPct      = Math.min(100, Math.round((verdictEligible / verdictThreshold) * 100))

  return (
    <div className="w-full max-w-4xl mx-auto space-y-6">
      <h2 className="text-lg font-semibold text-white">Training</h2>

      {error && (
        <div className="text-sm text-red-400 bg-red-900/30 border border-red-800 rounded px-3 py-2">
          {error}
        </div>
      )}

      {/* Offline training info banner */}
      <div className="bg-blue-900/15 border border-blue-800/50 rounded-lg p-4 flex gap-3">
        <span className="text-blue-400 text-lg leading-none mt-0.5">ℹ</span>
        <div className="space-y-1">
          <p className="text-sm font-medium text-blue-300">Initial training runs offline</p>
          <p className="text-xs text-gray-400 leading-relaxed">
            The classifier and autoencoder are trained on CIC-IDS2017 outside the UI to avoid
            long-running HTTP requests. Run <code className="text-gray-200 bg-gray-800 px-1 rounded">make train</code> on
            the host (requires the dataset in <code className="text-gray-200 bg-gray-800 px-1 rounded">data/CIC-IDS2017/</code>).
            Once training completes, the model appears in the history table below and the
            HITL fine-tune section unlocks.
          </p>
        </div>
      </div>

      {/* HITL fine-tune — only shown when an active model exists */}
      {activeModel ? (
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-5 space-y-4">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h3 className="text-sm font-semibold text-white">AE Fine-tune (HITL)</h3>
              <p className="text-xs text-gray-400 mt-1">
                Fine-tunes the <span className="text-gray-200">MLP Autoencoder only</span> on
                analyst-confirmed benign flows. Classifier weights stay frozen.
              </p>
              <p className="text-xs text-gray-500 mt-1">
                Active model: <span className="text-gray-300 font-mono">{activeModel.name}</span>
              </p>
            </div>
            <button
              onClick={handleRetrain}
              disabled={!retrainUnlocked || retraining}
              className="shrink-0 px-4 py-2 rounded text-sm font-medium bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {retraining ? 'Fine-tuning…' : 'Retrain AE'}
            </button>
          </div>

          {/* Verdict progress */}
          <div>
            <div className="flex justify-between text-xs text-gray-400 mb-1.5">
              <span>Analyst verdicts collected</span>
              <span>{verdictEligible} / {verdictThreshold}</span>
            </div>
            <div className="w-full bg-gray-800 rounded-full h-2">
              <div
                className={`h-2 rounded-full transition-all duration-500 ${retrainUnlocked ? 'bg-green-500' : 'bg-blue-500'}`}
                style={{ width: `${progressPct}%` }}
              />
            </div>
            {retrainUnlocked
              ? <p className="text-xs text-green-400 mt-1.5">Threshold reached — fine-tune unlocked</p>
              : <p className="text-xs text-gray-600 mt-1.5">Submit verdicts on alerts in the Inference tab to unlock</p>
            }
          </div>

          {/* SSE progress + log */}
          {retraining && (
            <div className="space-y-2">
              <div className="flex justify-between text-xs text-gray-400 mb-1">
                <span>Fine-tune progress</span><span>{pct}%</span>
              </div>
              <div className="w-full bg-gray-800 rounded-full h-1.5">
                <div className="bg-blue-500 h-1.5 rounded-full transition-all duration-300" style={{ width: `${pct}%` }} />
              </div>
            </div>
          )}
          {logs.length > 0 && (
            <div className="bg-gray-950 border border-gray-800 rounded p-3 max-h-36 overflow-y-auto font-mono text-xs text-gray-300 space-y-0.5">
              {logs.map((line, i) => <div key={i}>{line}</div>)}
              <div ref={logsEndRef} />
            </div>
          )}
          {retrainMsg && (
            <p className={`text-xs ${retrainMsg.ok ? 'text-green-400' : 'text-red-400'}`}>
              {retrainMsg.text}
            </p>
          )}
        </div>
      ) : (
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-5">
          <h3 className="text-sm font-semibold text-white mb-1">AE Fine-tune (HITL)</h3>
          <p className="text-sm text-gray-500">No active model — run initial training offline first.</p>
        </div>
      )}

      {/* Model version history */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
        <div className="px-5 py-3 border-b border-gray-800">
          <h3 className="text-sm font-medium text-gray-300">Model Version History</h3>
        </div>
        <table className="w-full text-sm">
          <thead>
            <tr className="text-xs text-gray-500 border-b border-gray-800">
              <th className="text-left px-5 py-3">Name</th>
              <th className="text-left px-4 py-3">Status</th>
              <th className="text-left px-4 py-3 cursor-help"
                title="F1-score — harmonic mean of Precision and Recall. Range 0–1; higher is better.">
                F1 <span className="text-gray-700">ⓘ</span>
              </th>
              <th className="text-left px-4 py-3 cursor-help"
                title="False Alarm Rate (FAR) — fraction of benign flows incorrectly flagged as attacks. Lower is better.">
                FAR <span className="text-gray-700">ⓘ</span>
              </th>
              <th className="text-left px-4 py-3 cursor-help"
                title="Attack Miss Rate (AMR) — fraction of true attack flows that escaped detection. Lower is better. 1.000 = AE threshold not yet calibrated.">
                Miss Rate ↓ <span className="text-gray-700">ⓘ</span>
              </th>
              <th className="text-left px-4 py-3">Created</th>
              <th className="px-4 py-3"></th>
            </tr>
          </thead>
          <tbody>
            {models.length === 0 ? (
              <tr>
                <td colSpan={7} className="px-5 py-6 text-center text-gray-500 text-sm">
                  No versions yet — run <code className="text-gray-400">make train</code> to create the first one.
                </td>
              </tr>
            ) : (
              models.map(m => (
                <tr key={m.id} className="border-b border-gray-800 hover:bg-gray-800/40">
                  <td className="px-5 py-3 text-white">{m.name}</td>
                  <td className="px-4 py-3">
                    <span className={`text-xs px-2 py-0.5 rounded ${STATUS_BADGE[m.status] ?? 'bg-gray-700 text-gray-300'}`}>
                      {m.status}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-gray-300 cursor-help"
                    title={m.metrics?.f1 != null ? `F1-score: ${m.metrics.f1.toFixed(6)} — harmonic mean of Precision and Recall` : undefined}>
                    {m.metrics?.f1?.toFixed(3) ?? '—'}
                  </td>
                  <td className="px-4 py-3 text-gray-300 cursor-help"
                    title={m.metrics?.far != null ? `False Alarm Rate: ${m.metrics.far.toFixed(6)} (${(m.metrics.far * 100).toFixed(4)}% of benign flows falsely flagged)` : undefined}>
                    {m.metrics?.far?.toFixed(5) ?? '—'}
                  </td>
                  <td className="px-4 py-3 text-gray-300 cursor-help"
                    title={m.metrics?.amr != null ? `Attack Miss Rate: ${m.metrics.amr.toFixed(6)} — ${(m.metrics.amr * 100).toFixed(1)}% of attacks escaped detection${m.metrics.amr >= 1.0 ? ' (AE threshold likely uncalibrated)' : ''}` : undefined}>
                    {m.metrics?.amr?.toFixed(3) ?? '—'}
                  </td>
                  <td className="px-4 py-3 text-gray-400">{new Date(m.created_at).toLocaleDateString()}</td>
                  <td className="px-4 py-3">
                    {m.status === 'archived' && (
                      <button onClick={() => handlePromote(m.id)} disabled={promotingId === m.id}
                        className="text-xs text-blue-400 hover:text-blue-300 disabled:opacity-40 disabled:cursor-not-allowed">
                        {promotingId === m.id ? '…' : 'Promote'}
                      </button>
                    )}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
