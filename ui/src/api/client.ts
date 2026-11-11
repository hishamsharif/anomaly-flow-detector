export interface HealthResponse {
  status: string
  model_loaded: boolean
  db_ok: boolean
}

export interface CheckResult {
  level: 'ok' | 'warn' | 'blocker'
  message: string
  detail?: string | null
}

export interface PreflightResponse {
  cuda: CheckResult
  docker: CheckResult
  images: CheckResult
}

async function request<T>(path: string): Promise<T> {
  const res = await fetch(path)
  if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`)
  return res.json() as Promise<T>
}

export const getHealth = () => request<HealthResponse>('/health')
export const getPreflight = () => request<PreflightResponse>('/api/preflight')

// ── Models ────────────────────────────────────────────────────────────────────

export interface TypeEntry {
  id: string
  label: string
}

export interface AvailableTypesResponse {
  classifiers: TypeEntry[]
  detectors: TypeEntry[]
}

export interface ModelVersionResponse {
  id: number
  name: string
  classifier_type: string
  detector_type: string
  status: string
  metrics: Record<string, number> | null
  created_at: string
}

export interface PaginatedModels {
  items: ModelVersionResponse[]
  total: number
}

export const getAvailableTypes = () => request<AvailableTypesResponse>('/api/models/available-types')
export const getActiveModel = () => request<ModelVersionResponse | null>('/api/models/active')
export const getModels = () => request<PaginatedModels>('/api/models')

// ── Training ──────────────────────────────────────────────────────────────────

export interface TrainingStartRequest {
  classifier_type: string
  detector_type: string
  mode?: string
}

export interface TrainingJobResponse {
  run_id: string
}

export interface TrainingStatusResponse {
  run_id: string
  pct: number
  mode: string
  started_at: string
}

export async function startTraining(req: TrainingStartRequest): Promise<TrainingJobResponse> {
  const res = await fetch('/api/training/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode: 'initial', ...req }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error((err as any).detail ?? `HTTP ${res.status}`)
  }
  return res.json() as Promise<TrainingJobResponse>
}

export function subscribeToTraining(run_id: string): EventSource {
  return new EventSource(`/api/training/${run_id}/events`)
}
