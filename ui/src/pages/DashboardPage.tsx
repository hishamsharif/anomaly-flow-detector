import { useEffect, useState } from 'react'
import { getActiveModel, getDashboardSummary, type DashboardSummary, type ModelVersionResponse } from '../api/client'

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

  if (error) return <p className="text-red-400 text-sm">{error}</p>
  if (!summary) return <p className="text-gray-500 text-sm animate-pulse">Loading…</p>

  const { by_severity: bs, verdict_eligible, verdict_threshold, retrain_unlocked } = summary
  const progressPct = verdict_threshold > 0
    ? Math.min(100, Math.round((verdict_eligible / verdict_threshold) * 100))
    : 100

  return (
    <div className="w-full max-w-4xl mx-auto space-y-6">
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
          <span className="text-xs font-medium text-amber-300 bg-amber-900/30 border border-amber-700/50 rounded-md px-3 py-1.5">
            ⚠ No active model — run <code className="font-mono">make train</code> first
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

      {/* HITL verdict progress — slim indicator; full retrain action is on Training tab */}
      <section className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <div className="flex items-center justify-between gap-4 mb-2">
          <div>
            <span className="text-xs font-medium text-gray-300">HITL verdict progress</span>
            <span className="text-xs text-gray-500 ml-2">
              {verdict_eligible} / {verdict_threshold} confirmed/override verdicts
            </span>
          </div>
          {retrain_unlocked && (
            <span className="text-xs font-medium text-green-400 bg-green-900/30 border border-green-700/50 rounded px-2 py-0.5">
              Fine-tune unlocked → Training tab
            </span>
          )}
        </div>
        <div className="w-full bg-gray-800 rounded-full h-1.5">
          <div
            className={`h-1.5 rounded-full transition-all duration-500 ${retrain_unlocked ? 'bg-green-500' : 'bg-blue-500'}`}
            style={{ width: `${progressPct}%` }}
          />
        </div>
        {!retrain_unlocked && (
          <p className="text-xs text-gray-600 mt-1.5">
            Submit verdicts on Inference alerts — once {verdict_threshold} are collected the AE can be fine-tuned from the Training tab.
          </p>
        )}
      </section>
    </div>
  )
}
