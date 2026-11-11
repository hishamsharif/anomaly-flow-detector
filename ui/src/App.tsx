import { useEffect, useState } from 'react'
import { Navigate, NavLink, Route, Routes } from 'react-router-dom'
import { type CheckResult, getHealth, getPreflight } from './api/client'
import AboutPage from './pages/AboutPage'
import DashboardPage from './pages/DashboardPage'
import InferencePage from './pages/InferencePage'
import TrainingPage from './pages/TrainingPage'

type ConnStatus = 'connecting' | 'connected' | 'error'

const TABS = ['dashboard', 'training', 'inference', 'about'] as const

export default function App() {
  const [connStatus, setConnStatus] = useState<ConnStatus>('connecting')
  const [blockers, setBlockers] = useState<CheckResult[]>([])

  useEffect(() => {
    getHealth()
      .then(() => setConnStatus('connected'))
      .catch(() => setConnStatus('error'))

    getPreflight()
      .then(data => setBlockers(Object.values(data).filter(c => c.level === 'blocker')))
      .catch(() => {})
  }, [])

  if (blockers.length > 0) {
    return (
      <div className="min-h-screen bg-gray-950 text-white flex items-center justify-center p-6">
        <div className="max-w-lg w-full">
          <h1 className="text-2xl font-bold mb-2 text-red-400">Prerequisites missing</h1>
          <p className="text-gray-400 mb-6">Fix the following before running the stack:</p>
          {blockers.map((b, i) => (
            <div key={i} className="bg-gray-900 border border-red-800 rounded-lg p-4 mb-3">
              <p className="font-semibold text-red-300">{b.message}</p>
              {b.detail && <p className="mt-1 text-sm text-gray-400">{b.detail}</p>}
            </div>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white flex flex-col">
      {/* Navbar */}
      <nav className="bg-gray-900 border-b border-gray-800 px-6 py-3 flex items-center gap-6">
        <span className="font-bold text-blue-400 mr-2 text-sm tracking-wider">AFD</span>
        {TABS.map(tab => (
          <NavLink
            key={tab}
            to={`/${tab}`}
            className={({ isActive }) =>
              `capitalize text-sm transition-colors ${
                isActive ? 'text-white font-medium' : 'text-gray-400 hover:text-gray-200'
              }`
            }
          >
            {tab}
          </NavLink>
        ))}

        {/* Connection badge */}
        <div className="ml-auto flex items-center gap-2 text-xs text-gray-400">
          <span
            className={`w-2 h-2 rounded-full ${
              connStatus === 'connected'
                ? 'bg-green-400'
                : connStatus === 'connecting'
                  ? 'bg-yellow-400 animate-pulse'
                  : 'bg-red-500'
            }`}
          />
          {connStatus === 'connected'
            ? 'Connected'
            : connStatus === 'connecting'
              ? 'Connecting…'
              : 'Disconnected'}
        </div>
      </nav>

      {/* Page content */}
      <main className="flex-1 p-6">
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/training" element={<TrainingPage />} />
          <Route path="/inference" element={<InferencePage />} />
          <Route path="/about" element={<AboutPage />} />
        </Routes>
      </main>
    </div>
  )
}
