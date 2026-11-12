import { useEffect, useState } from 'react'
import { getAbout, type AboutResponse } from '../api/client'

// ── Pipeline diagram ──────────────────────────────────────────────────────────

function PipelineDiagram() {
  const stages = [
    {
      num: '1',
      title: 'Rule Engine',
      desc: 'CVE signatures, IP blocklists, trusted ranges',
      color: 'border-purple-700 bg-purple-900/20',
      badge: 'bg-purple-800 text-purple-200',
    },
    {
      num: '2',
      title: 'XGBoost Classifier',
      desc: 'Supervised — detects known attack patterns',
      color: 'border-blue-700 bg-blue-900/20',
      badge: 'bg-blue-800 text-blue-200',
    },
    {
      num: '3',
      title: 'MLP Autoencoder',
      desc: 'Benign-only trained — high reconstruction error = anomaly',
      color: 'border-cyan-700 bg-cyan-900/20',
      badge: 'bg-cyan-800 text-cyan-200',
    },
  ]

  const verdicts = [
    { label: 'SAFE', cls: 'bg-green-900/60 text-green-300 border-green-700' },
    { label: 'KNOWN ATTACK', cls: 'bg-orange-900/60 text-orange-300 border-orange-700' },
    { label: 'NOVEL ATTACK', cls: 'bg-red-900/60 text-red-400 border-red-800' },
  ]

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold text-gray-300">3-Stage Detection Pipeline</h3>
      <div className="flex flex-col sm:flex-row items-stretch gap-2">
        {/* Input */}
        <div className="flex items-center justify-center sm:justify-start">
          <div className="text-xs text-gray-500 font-mono bg-gray-800 border border-gray-700 rounded px-3 py-2 whitespace-nowrap">
            Network Flow
          </div>
        </div>
        <div className="flex items-center text-gray-600 text-lg self-center">→</div>
        {/* Stages */}
        {stages.map((s, i) => (
          <div key={s.num} className="flex items-center gap-2">
            <div className={`flex-1 border rounded-lg p-3 min-w-[140px] ${s.color}`}>
              <div className="flex items-center gap-2 mb-1">
                <span className={`text-xs font-bold px-1.5 py-0.5 rounded ${s.badge}`}>
                  Stage {s.num}
                </span>
              </div>
              <p className="text-sm font-medium text-white">{s.title}</p>
              <p className="text-xs text-gray-400 mt-0.5">{s.desc}</p>
            </div>
            {i < stages.length - 1 && (
              <span className="text-gray-600 text-lg flex-shrink-0">→</span>
            )}
          </div>
        ))}
        <div className="flex items-center text-gray-600 text-lg self-center">→</div>
        {/* Verdicts */}
        <div className="flex flex-col gap-1 justify-center">
          {verdicts.map(v => (
            <span key={v.label} className={`text-xs px-2 py-0.5 rounded border font-medium ${v.cls}`}>
              {v.label}
            </span>
          ))}
        </div>
      </div>
      <p className="text-xs text-gray-600">
        Stage 1 verdict is final. Stage 2 triggers on confidence ≥ threshold. Stage 3 runs on all ambiguous flows.
      </p>
    </div>
  )
}

// ── Config table ──────────────────────────────────────────────────────────────

function ConfigTable({ config }: { config: Record<string, string> }) {
  const rows = Object.entries(config)
  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="text-xs text-gray-500 border-b border-gray-800">
          <th className="text-left px-3 py-1.5">Setting</th>
          <th className="text-left px-3 py-1.5">Value</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([k, v]) => (
          <tr key={k} className="border-b border-gray-800/50">
            <td className="px-3 py-1.5 font-mono text-xs text-gray-400">{k}</td>
            <td className="px-3 py-1.5 font-mono text-xs text-gray-200">{v}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

// ── Tech stack ────────────────────────────────────────────────────────────────

const STACK = [
  { layer: 'API', tech: 'FastAPI + Pydantic v2' },
  { layer: 'DB', tech: 'SQLite + SQLAlchemy async' },
  { layer: 'Classifier', tech: 'XGBoost (GPU)' },
  { layer: 'Detector', tech: 'PyTorch MLP-AE (GPU)' },
  { layer: 'Feature eng.', tech: 'scikit-learn + SMOTE' },
  { layer: 'Data loading', tech: 'DuckDB' },
  { layer: 'PCAP', tech: 'nfstream' },
  { layer: 'UI', tech: 'React 18 + Vite + Tailwind' },
]

function TechStack() {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
      {STACK.map(s => (
        <div key={s.layer} className="bg-gray-800/50 rounded px-3 py-2">
          <p className="text-xs text-gray-500">{s.layer}</p>
          <p className="text-sm text-gray-200">{s.tech}</p>
        </div>
      ))}
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function AboutPage() {
  const [data, setData] = useState<AboutResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getAbout()
      .then(setData)
      .catch((e: any) => setError(e.message ?? 'Failed to load'))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="max-w-5xl space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-white">About</h2>
        {data && (
          <span className="text-xs text-gray-500 font-mono">v{data.version}</span>
        )}
      </div>

      {/* Pipeline */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-5">
        <PipelineDiagram />
      </div>

      {/* Active model */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-5 space-y-2">
        <h3 className="text-sm font-semibold text-gray-300">Active Model</h3>
        {loading && <p className="text-sm text-gray-500">Loading…</p>}
        {error && <p className="text-sm text-red-400">{error}</p>}
        {!loading && !error && (
          data?.active_model ? (
            <div className="text-sm space-y-1">
              <div className="flex items-center gap-3">
                <span className="text-white font-medium">{data.active_model.name}</span>
                <span className="text-xs bg-green-900/60 text-green-300 px-2 py-0.5 rounded">active</span>
              </div>
              <div className="flex gap-4 text-xs text-gray-400">
                <span>Classifier: <span className="text-gray-200">{data.active_model.classifier_type}</span></span>
                <span>Detector: <span className="text-gray-200">{data.active_model.detector_type}</span></span>
                <span>Created: <span className="text-gray-200">{new Date(data.active_model.created_at).toLocaleDateString()}</span></span>
              </div>
              {data.active_model.metrics && Object.keys(data.active_model.metrics).length > 0 && (
                <div className="flex flex-wrap gap-3 mt-2">
                  {Object.entries(data.active_model.metrics).slice(0, 6).map(([k, v]) => (
                    <span key={k} className="text-xs text-gray-400 bg-gray-800 rounded px-2 py-0.5">
                      {k}: <span className="text-gray-200">{typeof v === 'number' ? v.toFixed(4) : v}</span>
                    </span>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <p className="text-sm text-gray-500">No active model — run training first.</p>
          )
        )}
      </div>

      {/* Config */}
      {data?.config && Object.keys(data.config).length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
          <div className="px-5 py-3 border-b border-gray-800">
            <h3 className="text-sm font-semibold text-gray-300">Runtime Configuration</h3>
          </div>
          <div className="p-3">
            <ConfigTable config={data.config} />
          </div>
        </div>
      )}

      {/* Tech stack */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-5 space-y-3">
        <h3 className="text-sm font-semibold text-gray-300">Tech Stack</h3>
        <TechStack />
      </div>
    </div>
  )
}
