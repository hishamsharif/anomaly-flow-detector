import { useEffect, useState } from 'react'
import { getActiveModel, getDashboardSummary, startTraining, type DashboardSummary, type ModelVersionResponse } from '../api/client'

function StatCard({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 flex flex-col gap-1">
      <span className="text-xs text-gray-400 uppercase tracking-wide">{label}</span>
      <span className={`text-2xl font-bold ${color}`}>{value}</span>
    </div>
  )
}

export default function DashboardPage() {
  const [summary, setSummary] = useState<DashboardSummary | null>(null)
  const [activeModel, setActiveModel] = useState<ModelVersionResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [retraining, setRetraining] = useState(false)
  const [retrainMsg, setRetrainMsg] = useState<{ text: string; ok: boolean } | null>(null)

  const load = () => {
    getDashboardSummary()
      .then(setSummary)
      .catch(() => setError('Failed to load dashboard — is the API running?'))
    getActiveModel().then(setActiveModel).catch(() => {})
  }

  useEffect(() => {
    load()
    const id = setInterval(load, 30_000)
    return () => clearInterval(id)
  }, [])

  const handleRetrain = async () => {
    setRetrainMsg(null)
    setRetraining(true)
    try {
      await startTraining({
        classifier_type: activeModel?.classifier_type ?? 'xgboost',
        detector_type:   activeModel?.detector_type   ?? 'mlp_ae',
        mode: 'ae_retrain',
      })
      setRetrainMsg({ text: 'AE retrain started — check the Training tab for progress.', ok: true })
    } catch (e: any) {
      setRetrainMsg({ text: e.message ?? 'Failed to start retrain', ok: false })
    } finally {
      setRetraining(false)
    }
  }

  if (error) return <p className="text-red-400 text-sm">{error}</p>
  if (!summary) return <p className="text-gray-500 text-sm animate-pulse">Loading…</p>

  const { by_severity: bs, verdict_eligible, verdict_threshold, retrain_unlocked } = summary
  const progressPct = verdict_threshold > 0
    ? Math.min(100, Math.round((verdict_eligible / verdict_threshold) * 100))
    : 100

  return (
    <div className="max-w-4xl space-y-6">
      <div className="flex items-start justify-between gap-4">
        <h2 className="text-lg font-semibold text-white">Dashboard</h2>
        {activeModel ? (
          <div className="flex items-center gap-2 text-xs bg-gray-900 border border-gray-800 rounded-md px-3 py-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-green-400 shrink-0" />
            <span className="text-gray-400">Active model:</span>
            <span className="text-white font-medium font-mono">{activeModel.name}</span>
            <span className="text-gray-600 ml-1">
              F1 {activeModel.metrics?.f1 != null ? activeModel.metrics.f1.toFixed(3) : '—'}
            </span>
          </div>
        ) : (
          <span className="text-xs text-gray-600 bg-gray-900 border border-gray-800 rounded-md px-3 py-1.5">
            No active model — run training first
          </span>
        )}
      </div>

      {/* Consensus cards — 24 h window */}
      <section>
        <h3 className="text-xs text-gray-500 uppercase tracking-wide mb-3">
          Alerts by verdict — last 24 h
        </h3>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <StatCard label="Total" value={summary.total_24h} color="text-white" />
          <StatCard label="Safe" value={summary.safe_24h} color="text-green-300" />
          <StatCard label="Known Attack" value={summary.known_attack_24h} color="text-orange-300" />
          <StatCard label="Novel Attack" value={summary.novel_attack_24h} color="text-red-400" />
        </div>
        {summary.pending_verdict_24h > 0 && (
          <p className="text-xs text-yellow-400 mt-2">
            {summary.pending_verdict_24h} alert{summary.pending_verdict_24h !== 1 ? 's' : ''} awaiting analyst verdict
          </p>
        )}
      </section>

      {/* Severity cards */}
      <section>
        <h3 className="text-xs text-gray-500 uppercase tracking-wide mb-3">Alerts by severity</h3>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <StatCard label="Low" value={bs.low} color="text-gray-400" />
          <StatCard label="Medium" value={bs.medium} color="text-yellow-300" />
          <StatCard label="High" value={bs.high} color="text-orange-300" />
          <StatCard label="Critical" value={bs.critical} color="text-red-400" />
        </div>
      </section>

      {/* Retrain eligibility */}
      <section className="bg-gray-900 border border-gray-800 rounded-lg p-5">
        <div className="flex items-start justify-between mb-4 gap-4">
          <div>
            <h3 className="text-sm font-medium text-white">AE Retrain Eligibility</h3>
            <p className="text-xs text-gray-400 mt-1">
              {verdict_eligible} of {verdict_threshold} confirmed/override verdicts collected.
            </p>
          </div>
          <button
            onClick={handleRetrain}
            disabled={!retrain_unlocked || retraining}
            className="shrink-0 px-4 py-2 rounded text-sm font-medium bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {retraining ? 'Starting…' : 'Retrain AE'}
          </button>
        </div>

        <div className="flex justify-between text-xs text-gray-400 mb-1">
          <span>Verdict progress</span>
          <span>{verdict_eligible} / {verdict_threshold}</span>
        </div>
        <div className="w-full bg-gray-800 rounded-full h-2">
          <div
            className={`h-2 rounded-full transition-all duration-500 ${retrain_unlocked ? 'bg-green-500' : 'bg-blue-500'}`}
            style={{ width: `${progressPct}%` }}
          />
        </div>
        {retrain_unlocked && (
          <p className="text-xs text-green-400 mt-2">Threshold reached — retrain unlocked</p>
        )}

        {retrainMsg && (
          <p className={`mt-3 text-xs ${retrainMsg.ok ? 'text-green-400' : 'text-red-400'}`}>
            {retrainMsg.text}
          </p>
        )}
      </section>
    </div>
  )
}
