import { useEffect, useRef, useState } from 'react'
import {
  getAvailableTypes,
  getModels,
  promoteModel,
  startTraining,
  subscribeToTraining,
  type AvailableTypesResponse,
  type ModelVersionResponse,
} from '../api/client'

const STATUS_BADGE: Record<string, string> = {
  active: 'bg-green-900 text-green-300',
  training: 'bg-yellow-900 text-yellow-300',
  failed: 'bg-red-900 text-red-300',
  archived: 'bg-gray-800 text-gray-400',
}

export default function TrainingPage() {
  const [types, setTypes] = useState<AvailableTypesResponse | null>(null)
  const [models, setModels] = useState<ModelVersionResponse[]>([])
  const [classifier, setClassifier] = useState('')
  const [detector, setDetector] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [promotingId, setPromotingId] = useState<number | null>(null)

  // Training state
  const [training, setTraining] = useState(false)
  const [pct, setPct] = useState(0)
  const [logs, setLogs] = useState<string[]>([])
  const esRef = useRef<EventSource | null>(null)
  const logsEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    getAvailableTypes()
      .then(data => {
        setTypes(data)
        if (data.classifiers.length) setClassifier(data.classifiers[0].id)
        if (data.detectors.length) setDetector(data.detectors[0].id)
      })
      .catch(() => setError('Failed to load model types'))

    getModels()
      .then(data => setModels(data.items))
      .catch(() => setError('Failed to load model history'))
  }, [])

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  const refreshModels = () =>
    getModels()
      .then(data => setModels(data.items))
      .catch(() => {})

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

  const handleStart = async () => {
    setError(null)
    setTraining(true)
    setPct(0)
    setLogs([])

    try {
      const { run_id } = await startTraining({ classifier_type: classifier, detector_type: detector })
      const es = subscribeToTraining(run_id)
      esRef.current = es

      es.addEventListener('progress', e => {
        setLogs(prev => [...prev, e.data])
      })
      es.addEventListener('step', e => {
        setPct(parseInt(e.data, 10))
      })
      es.addEventListener('done', e => {
        setLogs(prev => [...prev, `✓ ${e.data}`])
        setPct(100)
        es.close()
        esRef.current = null
        setTraining(false)
        refreshModels()
      })
      es.addEventListener('error', e => {
        const msg = (e as MessageEvent).data ?? 'Training failed'
        setError(msg)
        setLogs(prev => [...prev, `✗ ${msg}`])
        es.close()
        esRef.current = null
        setTraining(false)
      })
    } catch (e: any) {
      setError(e.message ?? 'Failed to start training')
      setTraining(false)
    }
  }

  return (
    <div className="max-w-4xl">
      <h2 className="text-lg font-semibold text-white mb-6">Training</h2>

      {error && (
        <div className="mb-4 text-sm text-red-400 bg-red-900/30 border border-red-800 rounded px-3 py-2">
          {error}
        </div>
      )}

      {/* Controls */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-5 mb-6">
        <div className="flex flex-wrap gap-4 items-end">
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-400">Classifier</label>
            <select
              value={classifier}
              onChange={e => setClassifier(e.target.value)}
              disabled={training}
              className="bg-gray-800 border border-gray-700 text-white text-sm rounded px-3 py-2 min-w-[200px] disabled:opacity-50"
            >
              {types?.classifiers.map(c => (
                <option key={c.id} value={c.id}>{c.label}</option>
              ))}
            </select>
          </div>

          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-400">Detector</label>
            <select
              value={detector}
              onChange={e => setDetector(e.target.value)}
              disabled={training}
              className="bg-gray-800 border border-gray-700 text-white text-sm rounded px-3 py-2 min-w-[200px] disabled:opacity-50"
            >
              {types?.detectors.map(d => (
                <option key={d.id} value={d.id}>{d.label}</option>
              ))}
            </select>
          </div>

          <button
            onClick={handleStart}
            disabled={training || !classifier || !detector}
            className="px-4 py-2 rounded text-sm font-medium bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {training ? 'Training…' : 'Start Training'}
          </button>
        </div>

        {/* Progress bar */}
        {training && (
          <div className="mt-4">
            <div className="flex justify-between text-xs text-gray-400 mb-1">
              <span>Progress</span>
              <span>{pct}%</span>
            </div>
            <div className="w-full bg-gray-800 rounded-full h-2">
              <div
                className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${pct}%` }}
              />
            </div>
          </div>
        )}

        {/* Log lines */}
        {logs.length > 0 && (
          <div className="mt-3 bg-gray-950 border border-gray-800 rounded p-3 max-h-40 overflow-y-auto font-mono text-xs text-gray-300 space-y-0.5">
            {logs.map((line, i) => <div key={i}>{line}</div>)}
            <div ref={logsEndRef} />
          </div>
        )}
      </div>

      {/* Version history */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
        <div className="px-5 py-3 border-b border-gray-800">
          <h3 className="text-sm font-medium text-gray-300">Model Version History</h3>
        </div>
        <table className="w-full text-sm">
          <thead>
            <tr className="text-xs text-gray-500 border-b border-gray-800">
              <th className="text-left px-5 py-3">Name</th>
              <th className="text-left px-4 py-3">Status</th>
              <th className="text-left px-4 py-3">F1</th>
              <th className="text-left px-4 py-3">FAR</th>
              <th className="text-left px-4 py-3">Miss Rate ↓</th>
              <th className="text-left px-4 py-3">Created</th>
              <th className="px-4 py-3"></th>
            </tr>
          </thead>
          <tbody>
            {models.length === 0 ? (
              <tr>
                <td colSpan={7} className="px-5 py-6 text-center text-gray-500 text-sm">
                  No versions yet — start training to create the first one.
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
                  <td className="px-4 py-3 text-gray-300">{m.metrics?.f1?.toFixed(3) ?? '—'}</td>
                  <td className="px-4 py-3 text-gray-300">{m.metrics?.far?.toFixed(5) ?? '—'}</td>
                  <td className="px-4 py-3 text-gray-300">{m.metrics?.amr?.toFixed(3) ?? '—'}</td>
                  <td className="px-4 py-3 text-gray-400">{new Date(m.created_at).toLocaleDateString()}</td>
                  <td className="px-4 py-3">
                    {m.status === 'archived' ? (
                      <button
                        onClick={() => handlePromote(m.id)}
                        disabled={promotingId === m.id}
                        className="text-xs text-blue-400 hover:text-blue-300 disabled:opacity-40 disabled:cursor-not-allowed"
                      >
                        {promotingId === m.id ? '…' : 'Promote'}
                      </button>
                    ) : (
                      <span className="text-xs text-gray-600">—</span>
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
