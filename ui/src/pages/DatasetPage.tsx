import { useEffect, useRef, useState } from 'react'
import { getDatasetSummary, type DatasetSummary } from '../api/client'

// ── Snapshot constants (CIC-IDS2017 baseline — used when make eda has not run) ─

const SNAP_CLASS_LABELS = [
  'BENIGN', 'DoS Hulk', 'PortScan', 'DDoS', 'DoS GoldenEye',
  'FTP-Patator', 'SSH-Patator', 'DoS slowloris', 'DoS Slowhttptest',
  'Bot', 'Web Attack Brute Force', 'Web Attack XSS',
  'Infiltration', 'Web Attack SQL Injection', 'Heartbleed',
]
const SNAP_CLASS_COUNTS = [
  2273097, 231073, 158930, 128027, 10293,
  7938, 5897, 5796, 5499, 1966, 1507, 652, 36, 21, 11,
]
const SNAP_DAYS = [
  'Monday', 'Tuesday', 'Wednesday', 'Thu AM', 'Thu PM',
  'Fri AM', 'Fri PM\n(DDoS)', 'Fri PM\n(Scan)',
]
const SNAP_BENIGN_BY_DAY = [529918, 432074, 440031, 168186, 288566, 189067, 97718, 127537]
const SNAP_ATTACK_TYPES = [
  'FTP-Patator', 'SSH-Patator', 'DoS Hulk', 'DoS GoldenEye',
  'DoS slowloris', 'DoS Slowhttptest', 'Heartbleed',
  'Web Brute Force', 'Web XSS', 'Web SQL Injection',
  'Infiltration', 'Bot', 'DDoS', 'PortScan',
]
const SNAP_ATTACK_BY_DAY: number[][] = [
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [7938,5897,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,231073,10293,5796,5499,11,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,1507,652,21,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,36,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,1966,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,128027,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,158930],
]
const SNAP_CORR_SHORT = [
  'Flow IAT Std', 'Flow IAT Mean', 'Fwd IAT Mean', 'Bwd IAT Max',
  'Flow IAT Max', 'Fwd IAT Max', 'Bwd IAT Mean', 'Fwd IAT Min',
  'Bwd IAT Min', 'Flow Duration', 'Fwd IAT Total', 'Bwd IAT Total',
  'Max Pkt Len', 'Bwd Pkt Max', 'Bwd Pkt Std',
  'Min Pkt Len', 'Fwd Pkt Min', 'Bwd Pkt Mean', 'Fwd Pkt Std', 'Dest Port',
]
const SNAP_CORR_MATRIX: number[][] = [
  [1.000,0.974,0.966,0.883,0.908,0.906,0.940,0.915,0.902,0.533,0.532,0.506,-0.125,-0.132,-0.151,0.342,0.338,-0.124,-0.101,-0.186],
  [0.974,1.000,0.989,0.822,0.821,0.822,0.952,0.958,0.929,0.479,0.479,0.469,-0.161,-0.173,-0.191,0.344,0.341,-0.167,-0.117,-0.172],
  [0.966,0.989,1.000,0.823,0.795,0.799,0.975,0.984,0.962,0.437,0.439,0.444,-0.166,-0.180,-0.201,0.365,0.361,-0.165,-0.119,-0.155],
  [0.883,0.822,0.823,1.000,0.908,0.901,0.851,0.760,0.779,0.615,0.609,0.660,-0.017,-0.008,-0.069,0.292,0.287,-0.018,-0.055,-0.194],
  [0.908,0.821,0.795,0.908,1.000,0.994,0.775,0.714,0.713,0.647,0.643,0.575,0.010,0.019,-0.027,0.265,0.261,0.027,-0.040,-0.209],
  [0.906,0.822,0.799,0.901,0.994,1.000,0.776,0.718,0.716,0.647,0.651,0.575,-0.002,0.005,-0.043,0.272,0.267,0.011,-0.045,-0.203],
  [0.940,0.952,0.975,0.851,0.775,0.776,1.000,0.959,0.983,0.434,0.435,0.469,-0.162,-0.176,-0.199,0.363,0.358,-0.164,-0.115,-0.151],
  [0.915,0.958,0.984,0.760,0.714,0.718,0.959,1.000,0.974,0.331,0.333,0.350,-0.180,-0.199,-0.220,0.372,0.368,-0.166,-0.125,-0.125],
  [0.902,0.929,0.962,0.779,0.713,0.716,0.983,0.974,1.000,0.331,0.334,0.360,-0.178,-0.197,-0.218,0.368,0.363,-0.163,-0.123,-0.124],
  [0.533,0.479,0.437,0.615,0.647,0.647,0.434,0.331,0.331,1.000,0.997,0.954,0.267,0.329,0.241,-0.019,-0.020,0.202,0.083,-0.243],
  [0.532,0.479,0.439,0.609,0.643,0.651,0.435,0.333,0.334,0.997,1.000,0.951,0.257,0.317,0.228,-0.013,-0.013,0.190,0.078,-0.239],
  [0.506,0.469,0.444,0.660,0.575,0.575,0.469,0.350,0.360,0.954,0.951,1.000,0.255,0.318,0.222,-0.008,-0.008,0.181,0.078,-0.229],
  [-0.125,-0.161,-0.166,-0.017,0.010,-0.002,-0.162,-0.180,-0.178,0.267,0.257,0.255,1.000,0.824,0.750,-0.315,-0.314,0.718,0.695,-0.192],
  [-0.132,-0.173,-0.180,-0.008,0.019,0.005,-0.176,-0.199,-0.197,0.329,0.317,0.318,0.824,1.000,0.916,-0.359,-0.357,0.901,0.299,-0.242],
  [-0.151,-0.191,-0.201,-0.069,-0.027,-0.043,-0.199,-0.220,-0.218,0.241,0.228,0.222,0.750,0.916,1.000,-0.412,-0.411,0.822,0.343,-0.250],
  [0.342,0.344,0.365,0.292,0.265,0.272,0.363,0.372,0.368,-0.019,-0.013,-0.008,-0.315,-0.359,-0.412,1.000,0.985,-0.271,-0.229,-0.179],
  [0.338,0.341,0.361,0.287,0.261,0.267,0.358,0.368,0.363,-0.020,-0.013,-0.008,-0.314,-0.357,-0.411,0.985,1.000,-0.271,-0.228,-0.169],
  [-0.124,-0.167,-0.165,-0.018,0.027,0.011,-0.164,-0.166,-0.163,0.202,0.190,0.181,0.718,0.901,0.822,-0.271,-0.271,1.000,0.205,-0.236],
  [-0.101,-0.117,-0.119,-0.055,-0.040,-0.045,-0.115,-0.125,-0.123,0.083,0.078,0.078,0.695,0.299,0.343,-0.229,-0.228,0.205,1.000,-0.078],
  [-0.186,-0.172,-0.155,-0.194,-0.209,-0.203,-0.151,-0.125,-0.124,-0.243,-0.239,-0.229,-0.192,-0.242,-0.250,-0.179,-0.169,-0.236,-0.078,1.000],
]

// ── Shared attack palette (stable order by index) ─────────────────────────────
const ATTACK_PALETTE = [
  '#4361ee', '#7209b7', '#f72585', '#b5179e', '#560bad', '#480ca8', '#3a0ca3',
  '#e63946', '#f4a261', '#2a9d8f', '#06d6a0', '#118ab2', '#073b4c', '#ef476f',
  '#d62828', '#a8dadc', '#457b9d', '#1d3557', '#e9c46a', '#264653',
]

// ── Derive chart data from API summary ───────────────────────────────────────

interface ChartData {
  classLabels: string[]
  classCounts: number[]
  days: string[]
  benignByDay: number[]
  attackTypes: string[]
  attackByDay: number[][]
  corrShort: string[]
  corrMatrix: number[][]
  totalFlows: number
  featureCounts?: DatasetSummary['feature_counts']
}

function snapshotData(): ChartData {
  const total = SNAP_CLASS_COUNTS.reduce((a, b) => a + b, 0)
  return {
    classLabels: SNAP_CLASS_LABELS,
    classCounts: SNAP_CLASS_COUNTS,
    days: SNAP_DAYS,
    benignByDay: SNAP_BENIGN_BY_DAY,
    attackTypes: SNAP_ATTACK_TYPES,
    attackByDay: SNAP_ATTACK_BY_DAY,
    corrShort: SNAP_CORR_SHORT,
    corrMatrix: SNAP_CORR_MATRIX,
    totalFlows: total,
  }
}

function fromApiSummary(s: DatasetSummary): ChartData {
  const classLabels = s.class_counts.map(c => c.label)
  const classCounts = s.class_counts.map(c => c.count)
  const totalFlows  = s.total_flows

  // Build ordered attack type list (by total count desc, benign excluded)
  const attackTotals: Record<string, number> = {}
  for (const day of s.days) {
    for (const [lbl, cnt] of Object.entries(day.attacks)) {
      attackTotals[lbl] = (attackTotals[lbl] ?? 0) + cnt
    }
  }
  const attackTypes = Object.entries(attackTotals)
    .sort(([, a], [, b]) => b - a)
    .map(([lbl]) => lbl)

  const days      = s.days.map(d => d.name)
  const benignByDay = s.days.map(d => d.benign)
  const attackByDay = s.days.map(d =>
    attackTypes.map(t => d.attacks[t] ?? 0)
  )

  return {
    classLabels,
    classCounts,
    days,
    benignByDay,
    attackTypes,
    attackByDay,
    corrShort: s.corr_features,
    corrMatrix: s.corr_matrix,
    totalFlows,
    featureCounts: s.feature_counts,
  }
}

// ── Canvas hook ───────────────────────────────────────────────────────────────

function useCanvas(
  drawFn: (ctx: CanvasRenderingContext2D, w: number, h: number, dark: boolean) => void,
  deps: unknown[],
) {
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return

    const dark = document.documentElement.getAttribute('data-theme') === 'dark'
      || (window.matchMedia('(prefers-color-scheme: dark)').matches
          && document.documentElement.getAttribute('data-theme') !== 'light')

    const render = () => {
      const dpr = window.devicePixelRatio || 1
      const w = canvas.clientWidth
      const h = canvas.clientHeight
      if (w === 0 || h === 0) return
      canvas.width = w * dpr
      canvas.height = h * dpr
      const ctx = canvas.getContext('2d')!
      ctx.scale(dpr, dpr)
      drawFn(ctx, w, h, dark)
    }

    render()
    const ro = new ResizeObserver(render)
    ro.observe(canvas)
    return () => ro.disconnect()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps])

  return ref
}

// ── Colour helpers ────────────────────────────────────────────────────────────

function corrColor(v: number, dark: boolean): string {
  if (v > 0) {
    const r = Math.round(230 * v + (dark ? 20 : 255) * (1 - v))
    const g = Math.round(57 * v + (dark ? 20 : 255) * (1 - v))
    const b = Math.round(70 * v + (dark ? 20 : 255) * (1 - v))
    return `rgb(${r},${g},${b})`
  } else {
    const t = -v
    const r = Math.round(59 * t + (dark ? 20 : 255) * (1 - t))
    const g = Math.round(91 * t + (dark ? 20 : 255) * (1 - t))
    const b = Math.round(219 * t + (dark ? 20 : 255) * (1 - t))
    return `rgb(${r},${g},${b})`
  }
}

// ── Chart 0: Class imbalance donut ────────────────────────────────────────────

function ClassImbalanceDonut({ classCounts, totalFlows }: { classCounts: number[]; totalFlows: number }) {
  const benignCount = classCounts[0] ?? 0
  const attackCount = totalFlows - benignCount
  const ref = useCanvas((ctx, w, h, dark) => {
    const BENIGN_COUNT = benignCount
    const ATTACK_COUNT = attackCount
    const TOTAL_FLOWS  = totalFlows
    const cx = w / 2
    const cy = h / 2
    const outerR = Math.min(cx, cy) - 8
    const innerR = outerR * 0.58

    const benignFrac  = BENIGN_COUNT / TOTAL_FLOWS
    const attackFrac  = ATTACK_COUNT / TOTAL_FLOWS
    const benignEnd   = -Math.PI / 2 + benignFrac * 2 * Math.PI

    // Benign arc
    ctx.beginPath()
    ctx.moveTo(cx, cy)
    ctx.arc(cx, cy, outerR, -Math.PI / 2, benignEnd)
    ctx.arc(cx, cy, innerR, benignEnd, -Math.PI / 2, true)
    ctx.closePath()
    ctx.fillStyle = 'rgba(59,91,219,0.82)'
    ctx.fill()

    // Attack arc
    ctx.beginPath()
    ctx.moveTo(cx, cy)
    ctx.arc(cx, cy, outerR, benignEnd, -Math.PI / 2 + 2 * Math.PI)
    ctx.arc(cx, cy, innerR, -Math.PI / 2 + 2 * Math.PI, benignEnd, true)
    ctx.closePath()
    ctx.fillStyle = 'rgba(230,57,70,0.82)'
    ctx.fill()

    // Thin separator ring
    ctx.beginPath()
    ctx.arc(cx, cy, outerR, 0, 2 * Math.PI)
    ctx.strokeStyle = dark ? '#1a1f2e' : '#f0f2f8'
    ctx.lineWidth = 2
    ctx.stroke()
    ctx.beginPath()
    ctx.arc(cx, cy, innerR, 0, 2 * Math.PI)
    ctx.stroke()

    // Center label
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillStyle = dark ? '#e2e8f0' : '#1a202c'
    ctx.font = `bold ${Math.round(outerR * 0.3)}px system-ui, sans-serif`
    ctx.fillText((benignFrac * 100).toFixed(1) + '%', cx, cy - outerR * 0.12)
    ctx.font = `${Math.round(outerR * 0.17)}px system-ui, sans-serif`
    ctx.fillStyle = dark ? '#9ba3c0' : '#4a5068'
    ctx.fillText('Benign', cx, cy + outerR * 0.15)

    // Legend
    const ly = h - 14
    const items = [
      { label: `Benign — ${(benignFrac * 100).toFixed(1)}% (${(BENIGN_COUNT / 1e6).toFixed(2)}M)`, color: 'rgba(59,91,219,0.82)' },
      { label: `Attacks — ${(attackFrac * 100).toFixed(1)}% (${(ATTACK_COUNT / 1e3).toFixed(0)}k)`, color: 'rgba(230,57,70,0.82)' },
    ]
    let lx = w / 2 - 170
    items.forEach(({ label, color }) => {
      ctx.fillStyle = color
      ctx.fillRect(lx, ly - 6, 10, 10)
      ctx.fillStyle = dark ? '#9ba3c0' : '#4a5068'
      ctx.font = '10px system-ui, sans-serif'
      ctx.textAlign = 'left'
      ctx.textBaseline = 'middle'
      ctx.fillText(label, lx + 13, ly)
      lx += ctx.measureText(label).width + 32
    })
  }, [benignCount, attackCount, totalFlows])

  return <canvas ref={ref} style={{ width: '100%', height: '220px', display: 'block' }} />
}

// ── Chart 1: Class distribution (horizontal log-scale bar) ───────────────────

function ClassDistributionChart({ classLabels, classCounts }: { classLabels: string[]; classCounts: number[] }) {
  const ref = useCanvas((ctx, w, h, dark) => {
    const fg2  = dark ? '#9ba3c0' : '#4a5068'
    const fg3  = dark ? '#5c6480' : '#8891aa'
    const grid = dark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.07)'

    const marginL = 182
    const marginR = 60
    const marginT = 12
    const marginB = 30
    const barH = 18
    const gap = 6
    const n = classLabels.length
    const neededH = marginT + marginB + n * (barH + gap)

    // resize canvas height if needed
    if (h < neededH) {
      ctx.canvas.style.height = neededH + 'px'
      return
    }

    const chartW = w - marginL - marginR
    const logMin = Math.log10(10)
    const logMax = Math.log10(3_000_000)
    const toX = (v: number) => marginL + ((Math.log10(Math.max(v, 10)) - logMin) / (logMax - logMin)) * chartW

    // Grid lines at powers of 10
    const ticks = [100, 1_000, 10_000, 100_000, 1_000_000]
    ctx.strokeStyle = grid
    ctx.lineWidth = 1
    ticks.forEach(t => {
      const x = toX(t)
      ctx.beginPath(); ctx.moveTo(x, marginT); ctx.lineTo(x, h - marginB); ctx.stroke()
    })

    // Tick labels
    ctx.fillStyle = fg3
    ctx.font = '10px system-ui, sans-serif'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'top'
    ticks.forEach(t => {
      const label = t >= 1_000_000 ? (t / 1_000_000) + 'M' : t >= 1_000 ? (t / 1_000) + 'k' : String(t)
      ctx.fillText(label, toX(t), h - marginB + 4)
    })

    // Bars + labels
    classLabels.forEach((label, i) => {
      const count = classCounts[i]
      const y = marginT + i * (barH + gap)
      const barW = toX(count) - marginL

      ctx.fillStyle = label === 'BENIGN' ? 'rgba(59,91,219,0.75)' : 'rgba(230,57,70,0.72)'
      ctx.beginPath()
      ctx.roundRect(marginL, y, barW, barH, 2)
      ctx.fill()

      // Row label
      ctx.fillStyle = fg2
      ctx.font = '11px system-ui, sans-serif'
      ctx.textAlign = 'right'
      ctx.textBaseline = 'middle'
      ctx.fillText(label, marginL - 6, y + barH / 2)

      // Count label
      ctx.fillStyle = fg3
      ctx.font = '10px system-ui, sans-serif'
      ctx.textAlign = 'left'
      ctx.textBaseline = 'middle'
      const countStr = count >= 1_000_000
        ? (count / 1_000_000).toFixed(2) + 'M'
        : count >= 1_000 ? (count / 1_000).toFixed(1) + 'k' : String(count)
      ctx.fillText(countStr, marginL + barW + 4, y + barH / 2)
    })
  }, [classLabels, classCounts])

  return (
    <canvas
      ref={ref}
      style={{ width: '100%', height: `${classLabels.length * 24 + 42}px`, display: 'block' }}
    />
  )
}

// ── Chart 2: Stacked bar — attacks by capture day ────────────────────────────

function AttackByDayChart({ days, benignByDay, attackTypes, attackByDay }: {
  days: string[]
  benignByDay: number[]
  attackTypes: string[]
  attackByDay: number[][]
}) {
  const ref = useCanvas((ctx, w, h, dark) => {
    const fg2  = dark ? '#9ba3c0' : '#4a5068'
    const fg3  = dark ? '#5c6480' : '#8891aa'
    const grid = dark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.07)'
    const benignColor = dark ? 'rgba(59,91,219,0.35)' : 'rgba(59,91,219,0.2)'
    const benignBorder = 'rgba(59,91,219,0.6)'

    const marginL = 48
    const marginR = 16
    const marginT = 12
    const marginB = 46
    const chartH = h - marginT - marginB
    const chartW = w - marginL - marginR

    const nDays = days.length
    const barW = Math.floor((chartW / nDays) * 0.65)
    const barSpacing = chartW / nDays

    // Max stack height
    const maxVal = Math.max(
      ...days.map((_, di) => {
        const benign = benignByDay[di]
        const attacks = attackByDay[di].reduce((a, b) => a + b, 0)
        return benign + attacks
      })
    )
    const toY = (v: number) => marginT + chartH - (v / maxVal) * chartH

    // Y grid + labels
    const yTicks = [0, 100_000, 200_000, 300_000, 400_000, 500_000, 600_000, 700_000].filter(t => t <= maxVal * 1.05)
    ctx.strokeStyle = grid
    ctx.lineWidth = 1
    yTicks.forEach(t => {
      const y = toY(t)
      ctx.beginPath(); ctx.moveTo(marginL, y); ctx.lineTo(w - marginR, y); ctx.stroke()
      ctx.fillStyle = fg3
      ctx.font = '9px system-ui, sans-serif'
      ctx.textAlign = 'right'
      ctx.textBaseline = 'middle'
      ctx.fillText(t >= 1000 ? (t / 1000) + 'k' : String(t), marginL - 4, y)
    })

    // Bars
    days.forEach((day, di) => {
      const cx = marginL + di * barSpacing + barSpacing / 2
      const x = cx - barW / 2

      // Benign (bottom)
      const benign = benignByDay[di]
      const benignH = (benign / maxVal) * chartH
      ctx.fillStyle = benignColor
      ctx.strokeStyle = benignBorder
      ctx.lineWidth = 0.5
      ctx.beginPath()
      ctx.roundRect(x, toY(benign), barW, benignH, [0, 0, 2, 2])
      ctx.fill(); ctx.stroke()

      // Attack types (stacked above benign)
      let stackBase = benign
      attackByDay[di].forEach((count, ai) => {
        if (count === 0) return
        const y0 = toY(stackBase + count)
        const segH = (count / maxVal) * chartH
        ctx.fillStyle = ATTACK_PALETTE[ai % ATTACK_PALETTE.length] + 'cc'
        ctx.strokeStyle = ATTACK_PALETTE[ai % ATTACK_PALETTE.length]
        ctx.lineWidth = 0.5
        ctx.beginPath()
        ctx.roundRect(x, y0, barW, segH, 1)
        ctx.fill(); ctx.stroke()
        stackBase += count
      })

      // Day label
      ctx.fillStyle = fg2
      ctx.font = '10px system-ui, sans-serif'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'top'
      const lines = day.split('\n')
      lines.forEach((line, li) => ctx.fillText(line, cx, h - marginB + 4 + li * 12))
    })

    // Legend — attack types with nonzero total
    const legendTypes = attackTypes.map((name, i) => ({ name, color: ATTACK_PALETTE[i % ATTACK_PALETTE.length] }))
      .filter((_, i) => attackByDay.some(d => d[i] > 0))

    let lx = marginL, ly = h - marginB + 32
    ctx.font = '9px system-ui, sans-serif'
    ctx.textBaseline = 'middle'
    legendTypes.forEach(({ name, color }) => {
      if (lx + 90 > w - marginR) { lx = marginL; ly += 14 }
      ctx.fillStyle = color + 'cc'
      ctx.fillRect(lx, ly - 4, 8, 8)
      ctx.fillStyle = fg3
      ctx.textAlign = 'left'
      ctx.fillText(name, lx + 10, ly)
      lx += ctx.measureText(name).width + 20
    })

    // Benign legend item
    ctx.fillStyle = benignColor
    ctx.strokeStyle = benignBorder
    ctx.lineWidth = 0.5
    ctx.beginPath(); ctx.rect(marginL, ly + 10 - 4, 8, 8); ctx.fill(); ctx.stroke()
    ctx.fillStyle = fg3; ctx.textAlign = 'left'
    ctx.fillText('BENIGN', marginL + 10, ly + 10)
  }, [days, benignByDay, attackTypes, attackByDay])

  return <canvas ref={ref} style={{ width: '100%', height: '320px', display: 'block' }} />
}

// ── Chart 3: Correlation heatmap ─────────────────────────────────────────────

function CorrelationHeatmap({ corrShort, corrMatrix }: { corrShort: string[]; corrMatrix: number[][] }) {
  const ref = useCanvas((ctx, w, _h, dark) => {
    const n = corrShort.length
    const fg3 = dark ? '#5c6480' : '#8891aa'
    const labelFont = '9px system-ui, sans-serif'

    const labelW = 72
    const labelH = 70
    const maxCell = Math.floor((w - labelW - 4) / n)
    const cellSize = Math.min(maxCell, 36)

    // Cells
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        const v = corrMatrix[r]?.[c] ?? 0
        const x = labelW + c * cellSize
        const y = labelH + r * cellSize
        ctx.fillStyle = corrColor(v, dark)
        ctx.fillRect(x, y, cellSize - 1, cellSize - 1)

        if (cellSize >= 28) {
          const textBright = Math.abs(v) > 0.6
          ctx.fillStyle = textBright ? (dark ? '#fff' : '#fff') : (dark ? '#bbb' : '#444')
          ctx.font = '8px system-ui, sans-serif'
          ctx.textAlign = 'center'
          ctx.textBaseline = 'middle'
          ctx.fillText(v.toFixed(2), x + cellSize / 2 - 0.5, y + cellSize / 2)
        }
      }
    }

    // Row labels
    ctx.fillStyle = fg3
    ctx.font = labelFont
    ctx.textAlign = 'right'
    ctx.textBaseline = 'middle'
    for (let r = 0; r < n; r++) {
      ctx.fillText(corrShort[r], labelW - 4, labelH + r * cellSize + cellSize / 2)
    }

    // Column labels (rotated)
    for (let c = 0; c < n; c++) {
      const x = labelW + c * cellSize + cellSize / 2
      ctx.save()
      ctx.translate(x, labelH - 4)
      ctx.rotate(-Math.PI / 2.5)
      ctx.fillStyle = fg3
      ctx.font = labelFont
      ctx.textAlign = 'left'
      ctx.textBaseline = 'middle'
      ctx.fillText(corrShort[c], 0, 0)
      ctx.restore()
    }

    // Color scale legend bar
    const barY = labelH + n * cellSize + 6
    const barX = labelW
    const barW2 = n * cellSize
    const barH2 = 8
    const grad = ctx.createLinearGradient(barX, 0, barX + barW2, 0)
    grad.addColorStop(0, corrColor(-1, dark))
    grad.addColorStop(0.5, corrColor(0, dark))
    grad.addColorStop(1, corrColor(1, dark))
    ctx.fillStyle = grad
    ctx.fillRect(barX, barY, barW2, barH2)
    ctx.fillStyle = fg3
    ctx.font = '8px system-ui, sans-serif'
    ctx.textAlign = 'left'; ctx.textBaseline = 'top'
    ctx.fillText('−1.0', barX, barY + barH2 + 2)
    ctx.textAlign = 'center'
    ctx.fillText('0', barX + barW2 / 2, barY + barH2 + 2)
    ctx.textAlign = 'right'
    ctx.fillText('+1.0', barX + barW2, barY + barH2 + 2)
  }, [corrShort, corrMatrix])

  return (
    <canvas
      ref={ref}
      style={{ width: '800px', minWidth: '800px', height: `${corrShort.length * 36 + 90}px`, display: 'block' }}
    />
  )
}

// ── Insight card ──────────────────────────────────────────────────────────────

function Insight({ adr, children }: { adr?: string; children: React.ReactNode }) {
  return (
    <div className="rounded-md border border-blue-900/40 bg-blue-900/10 px-4 py-3 text-sm text-gray-300 space-y-1 mt-4">
      {adr && (
        <span className="inline-block text-xs font-semibold text-blue-400 uppercase tracking-wider mb-1">
          {adr}
        </span>
      )}
      <div className="leading-relaxed">{children}</div>
    </div>
  )
}

// ── Stat chips ────────────────────────────────────────────────────────────────

function Stat({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div className={`rounded-md border px-4 py-2.5 ${accent ?? 'border-gray-800 bg-gray-900'}`}>
      <p className="text-xs text-gray-500 mb-0.5">{label}</p>
      <p className="text-lg font-semibold text-white font-mono tabular-nums">{value}</p>
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function DatasetPage() {
  const [apiSummary, setApiSummary] = useState<DatasetSummary | null>(null)
  const [isSnapshot, setIsSnapshot] = useState(true)

  useEffect(() => {
    getDatasetSummary()
      .then(s => { setApiSummary(s); setIsSnapshot(false) })
      .catch(() => setIsSnapshot(true))
  }, [])

  const data: ChartData = apiSummary ? fromApiSummary(apiSummary) : snapshotData()

  const totalFlows   = data.totalFlows
  const benignCount  = data.classCounts[0] ?? 0
  const attackClasses = data.classLabels.filter(l => l !== 'BENIGN').length
  const rawFeatures  = data.featureCounts?.raw ?? 79
  const benignPct    = totalFlows > 0 ? (benignCount / totalFlows * 100).toFixed(1) : '—'

  const fmtFlows = (n: number) =>
    n >= 1_000_000 ? `${(n / 1_000_000).toFixed(2)}M` : n >= 1_000 ? `${(n / 1_000).toFixed(0)}k` : String(n)

  return (
    <div className="w-full max-w-5xl mx-auto space-y-8">

      {/* Header */}
      <div>
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold text-white">Dataset — CIC-IDS2017</h2>
            <p className="text-sm text-gray-400 mt-1">
              Exploratory data analysis of the training corpus. Each finding below informed a specific
              architectural decision in the AFD V2 pipeline.
            </p>
          </div>
          {isSnapshot ? (
            <span className="shrink-0 text-xs text-gray-500 bg-gray-900 border border-gray-800 rounded px-2.5 py-1.5 leading-tight text-right">
              Snapshot data<br />
              <code className="text-gray-400">make eda</code> to refresh
            </span>
          ) : (
            <span className="shrink-0 text-xs text-green-400 bg-green-900/20 border border-green-800/40 rounded px-2.5 py-1.5 leading-tight text-right">
              Live EDA data<br />
              {new Date(apiSummary!.generated_at).toLocaleDateString()}
            </span>
          )}
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
          <Stat label="Total flows" value={fmtFlows(totalFlows)} />
          <Stat label="Benign" value={`${benignPct}%`} accent="border-blue-900/40 bg-blue-900/10" />
          <Stat label="Attack classes" value={String(attackClasses)} accent="border-red-900/40 bg-red-900/10" />
          <Stat label="Raw features" value={String(rawFeatures)} />
        </div>

        {/* Preprocessing pipeline counts — only shown when EDA has been run */}
        {data.featureCounts && (
          <div className="mt-3 flex flex-wrap gap-2 text-xs text-gray-500">
            <span className="bg-gray-900 border border-gray-800 rounded px-2 py-1">
              Raw <span className="text-white font-mono">{data.featureCounts.raw}</span>
            </span>
            <span className="text-gray-700 self-center">→</span>
            <span className="bg-gray-900 border border-gray-800 rounded px-2 py-1">
              After variance <span className="text-white font-mono">{data.featureCounts.after_variance}</span>
            </span>
            <span className="text-gray-700 self-center">→</span>
            <span className="bg-gray-900 border border-gray-800 rounded px-2 py-1">
              After correlation <span className="text-white font-mono">{data.featureCounts.after_correlation}</span>
            </span>
            <span className="text-gray-700 self-center">→</span>
            <span className="bg-gray-900 border border-green-900/50 rounded px-2 py-1">
              Selected <span className="text-green-400 font-mono">{data.featureCounts.after_selector}</span>
            </span>
          </div>
        )}
      </div>

      {/* Figure 1 */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-5 space-y-4">
        <div>
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Figure 1</p>
          <h3 className="text-sm font-semibold text-white mt-0.5">
            Class distribution — {attackClasses} attack types
          </h3>
          <p className="text-xs text-gray-500 mt-0.5">Log-scale x-axis · full {fmtFlows(totalFlows)}-flow corpus · no sampling</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-[220px_1fr] gap-6 items-start">
          <div>
            <p className="text-xs text-gray-500 text-center mb-1 uppercase tracking-wide">Binary split</p>
            <ClassImbalanceDonut classCounts={data.classCounts} totalFlows={data.totalFlows} />
          </div>
          <div>
            <p className="text-xs text-gray-500 mb-1 uppercase tracking-wide">Per-class breakdown (log scale)</p>
            <div className="overflow-x-auto">
              <ClassDistributionChart classLabels={data.classLabels} classCounts={data.classCounts} />
            </div>
          </div>
        </div>

        <Insight adr="Motivates ADR-04 & ADR-14">
          <strong className="text-white">Why plain accuracy is misleading:</strong>{' '}
          A classifier that always predicts BENIGN would score <strong className="text-white">{benignPct}% accuracy</strong> without
          detecting a single attack. This imbalance is why the pipeline uses{' '}
          <span className="text-blue-300">F1-score, Precision/Recall and ROC-AUC</span> as primary metrics, and
          applies <span className="text-blue-300">SMOTE oversampling</span> (capped per class) on the training split only.
        </Insight>
      </div>

      {/* Figure 2 */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-5 space-y-4">
        <div>
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Figure 2</p>
          <h3 className="text-sm font-semibold text-white mt-0.5">Attack types by capture day</h3>
          <p className="text-xs text-gray-500 mt-0.5">Blue = benign flows · coloured segments = attack types · each day introduces new, unseen classes</p>
        </div>

        <div className="overflow-x-auto">
          <AttackByDayChart
            days={data.days}
            benignByDay={data.benignByDay}
            attackTypes={data.attackTypes}
            attackByDay={data.attackByDay}
          />
        </div>

        <Insight adr="Motivates ADR-02 (day-based temporal split)">
          <strong className="text-white">Why a random split would flatter the model:</strong>{' '}
          Monday is 100% benign. Each subsequent day introduces attack types not seen before —
          Tuesday adds FTP/SSH brute-force attacks, Wednesday adds four DoS variants,
          Thursday adds three web-attack types, Friday brings Bot, DDoS and PortScan.
          A random 80/20 split leaks examples of <em>every</em> attack type into training, inflating held-out performance.
          The <span className="text-blue-300">temporal split (Mon–Thu → train, Fri-AM → val, Fri-PM → test)</span> forces
          the model to generalise to attack patterns it has never seen — the same challenge faced in production.
        </Insight>
      </div>

      {/* Figure 3 */}
      {data.corrShort.length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-5 space-y-4">
          <div>
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Figure 3</p>
            <h3 className="text-sm font-semibold text-white mt-0.5">
              Feature correlation heatmap — top-{data.corrShort.length} features by mean |ρ|
            </h3>
            <p className="text-xs text-gray-500 mt-0.5">
              Pearson ρ · {BENIGN_SAMPLE_N.toLocaleString()} benign flows sampled · red = strong positive · blue = strong negative
            </p>
          </div>

          <div className="overflow-x-auto">
            <CorrelationHeatmap corrShort={data.corrShort} corrMatrix={data.corrMatrix} />
          </div>

          <Insight adr="Motivates ADR-10 (correlation filter)">
            <strong className="text-white">Two redundant feature clusters:</strong>{' '}
            IAT features (Flow/Fwd/Bwd Inter-Arrival Times) form a tightly correlated block with ρ ≈ 0.95–0.99 —
            keeping all of them adds noise without information.
            Packet-length features form a second cluster at ρ ≈ 0.75–0.92.
            The <span className="text-blue-300">correlation pruning step</span> drops one feature from each pair
            where ρ {'>'} threshold, reducing raw features to ~{data.featureCounts?.after_correlation ?? '~40'} without
            measurable loss in classifier performance or AE reconstruction fidelity.
          </Insight>
        </div>
      )}

    </div>
  )
}

const BENIGN_SAMPLE_N = 5_000
