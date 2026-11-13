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

// ── Alerts + Inference ────────────────────────────────────────────────────────

export interface FeatureContribution {
  name: string
  contribution: number
}

export interface AlertResponse {
  id: number
  flow_id: string
  consensus: string
  severity: string
  predicted_label: number | null
  classifier_confidence: number | null
  ae_score: number
  ae_threshold: number
  top_features: FeatureContribution[]
  analyst_verdict: string | null
  feedback_at: string | null
  timestamp: string
}

export interface RejectedRow {
  row: number
  reason: string
}

export interface BatchSummary {
  total: number
  safe: number
  known_attack: number
  novel_attack: number
  rejected: number
}

export interface BatchPredictResponse {
  alerts: AlertResponse[]
  rejected_rows: RejectedRow[]
  summary: BatchSummary
}

export interface PaginatedAlerts {
  items: AlertResponse[]
  total: number
  page: number
  size: number
}

export type PredictResult =
  | { kind: 'sync'; data: BatchPredictResponse }
  | { kind: 'async'; job_id: string; flow_count: number }

export async function batchPredict(file: File): Promise<PredictResult> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch('/api/predict/batch', { method: 'POST', body: form })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error((err as any).detail ?? `HTTP ${res.status}`)
  }
  if (res.status === 202) {
    const data = await res.json()
    return { kind: 'async', ...data }
  }
  const data = await res.json()
  return { kind: 'sync', data }
}

export const getPredictJobStatus = (job_id: string) =>
  request<{ job_id: string; pct: number; done: boolean }>(`/api/predict/${job_id}/status`)

export const getPredictJobResults = (job_id: string) =>
  request<BatchPredictResponse>(`/api/predict/${job_id}/results`)

export const getAlerts = (page = 1, size = 50, consensus?: string, verdict?: string) => {
  const params = new URLSearchParams({ page: String(page), size: String(size) })
  if (consensus) params.set('consensus', consensus)
  if (verdict) params.set('verdict', verdict)
  return request<PaginatedAlerts>(`/api/alerts?${params}`)
}

export async function submitVerdict(alertId: number, verdict: string, label?: string): Promise<AlertResponse> {
  const res = await fetch(`/api/alerts/${alertId}/verdict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ verdict, label }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error((err as any).detail ?? `HTTP ${res.status}`)
  }
  return res.json() as Promise<AlertResponse>
}

export const getVerdictCount = () =>
  request<{ eligible: number; threshold: number; unlocked: boolean }>('/api/alerts/verdict-count')

// ── About ─────────────────────────────────────────────────────────────────────

export interface AboutResponse {
  version: string
  active_model: ModelVersionResponse | null
  config: Record<string, string>
}

export const getAbout = () => request<AboutResponse>('/api/about')

// ── Dashboard ─────────────────────────────────────────────────────────────────

export interface DashboardSummary {
  total_24h: number
  safe_24h: number
  known_attack_24h: number
  novel_attack_24h: number
  pending_verdict_24h: number
  by_severity: { low: number; medium: number; high: number; critical: number }
  verdict_eligible: number
  verdict_threshold: number
  retrain_unlocked: boolean
}

export const getDashboardSummary = () => request<DashboardSummary>('/api/dashboard/summary')

// ── Dataset EDA ───────────────────────────────────────────────────────────────

export interface ClassCount {
  label: string
  count: number
}

export interface DayBreakdown {
  name: string
  split: string
  benign: number
  attacks: Record<string, number>
}

export interface FeatureCounts {
  raw: number
  after_variance: number
  after_correlation: number
  after_selector: number
}

export interface DatasetSummary {
  total_flows: number
  class_counts: ClassCount[]
  days: DayBreakdown[]
  corr_features: string[]
  corr_matrix: number[][]
  feature_counts: FeatureCounts
  generated_at: string
}

export const getDatasetSummary = () => request<DatasetSummary>('/api/dataset/summary')

export async function promoteModel(id: number): Promise<ModelVersionResponse> {
  const res = await fetch(`/api/models/${id}/promote`, { method: 'POST' })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error((err as any).detail ?? `HTTP ${res.status}`)
  }
  return res.json() as Promise<ModelVersionResponse>
}
