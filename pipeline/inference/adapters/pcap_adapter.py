from __future__ import annotations

import os
import tempfile

# ── CIC-IDS2017 column → nfstream row extractor ───────────────────────────────
# nfstream `statistical_analysis=True` row is a pandas Series.
# Unmapped or erroring features fall through to the default of 0.0.

def _ms_to_us(ms: float) -> float:
    return ms * 1000.0

def _rate(numerator: float, duration_ms: float) -> float:
    return numerator / (duration_ms / 1000.0) if duration_ms > 0 else 0.0


_EXTRACTORS: dict[str, object] = {
    "Flow Duration":               lambda r: _ms_to_us(r.bidirectional_duration_ms),
    "Total Fwd Packets":           lambda r: r.src2dst_packets,
    "Total Backward Packets":      lambda r: r.dst2src_packets,
    "Total Length of Fwd Packets": lambda r: r.src2dst_bytes,
    "Total Length of Bwd Packets": lambda r: r.dst2src_bytes,
    "Fwd Packet Length Max":       lambda r: r.src2dst_max_ps,
    "Fwd Packet Length Min":       lambda r: r.src2dst_min_ps,
    "Fwd Packet Length Mean":      lambda r: r.src2dst_mean_ps,
    "Fwd Packet Length Std":       lambda r: r.src2dst_stddev_ps,
    "Bwd Packet Length Max":       lambda r: r.dst2src_max_ps,
    "Bwd Packet Length Min":       lambda r: r.dst2src_min_ps,
    "Bwd Packet Length Mean":      lambda r: r.dst2src_mean_ps,
    "Bwd Packet Length Std":       lambda r: r.dst2src_stddev_ps,
    "Flow Bytes/s":    lambda r: _rate(r.bidirectional_bytes, r.bidirectional_duration_ms),
    "Flow Packets/s":  lambda r: _rate(r.bidirectional_packets, r.bidirectional_duration_ms),
    "Flow IAT Mean":   lambda r: _ms_to_us(r.bidirectional_mean_piat_ms),
    "Flow IAT Std":    lambda r: _ms_to_us(r.bidirectional_stddev_piat_ms),
    "Flow IAT Max":    lambda r: _ms_to_us(r.bidirectional_max_piat_ms),
    "Flow IAT Min":    lambda r: _ms_to_us(r.bidirectional_min_piat_ms),
    "Fwd IAT Total":   lambda r: _ms_to_us(r.src2dst_duration_ms),
    "Fwd IAT Mean":    lambda r: _ms_to_us(r.src2dst_mean_piat_ms),
    "Fwd IAT Std":     lambda r: _ms_to_us(r.src2dst_stddev_piat_ms),
    "Fwd IAT Max":     lambda r: _ms_to_us(r.src2dst_max_piat_ms),
    "Fwd IAT Min":     lambda r: _ms_to_us(r.src2dst_min_piat_ms),
    "Bwd IAT Total":   lambda r: _ms_to_us(r.dst2src_duration_ms),
    "Bwd IAT Mean":    lambda r: _ms_to_us(r.dst2src_mean_piat_ms),
    "Bwd IAT Std":     lambda r: _ms_to_us(r.dst2src_stddev_piat_ms),
    "Bwd IAT Max":     lambda r: _ms_to_us(r.dst2src_max_piat_ms),
    "Bwd IAT Min":     lambda r: _ms_to_us(r.dst2src_min_piat_ms),
    "Fwd PSH Flags":           lambda r: r.src2dst_psh_packets,
    "Bwd PSH Flags":           lambda r: r.dst2src_psh_packets,
    "Fwd URG Flags":           lambda r: r.src2dst_urg_packets,
    "Bwd URG Flags":           lambda r: r.dst2src_urg_packets,
    "Min Packet Length":       lambda r: r.bidirectional_min_ps,
    "Max Packet Length":       lambda r: r.bidirectional_max_ps,
    "Packet Length Mean":      lambda r: r.bidirectional_mean_ps,
    "Packet Length Std":       lambda r: r.bidirectional_stddev_ps,
    "Packet Length Variance":  lambda r: r.bidirectional_stddev_ps ** 2,
    "FIN Flag Count": lambda r: r.src2dst_fin_packets + r.dst2src_fin_packets,
    "SYN Flag Count": lambda r: r.bidirectional_syn_packets,
    "RST Flag Count": lambda r: r.bidirectional_rst_packets,
    "PSH Flag Count": lambda r: r.bidirectional_psh_packets,
    "ACK Flag Count": lambda r: r.bidirectional_ack_packets,
    "URG Flag Count": lambda r: r.bidirectional_urg_packets,
    "CWE Flag Count": lambda r: r.bidirectional_cwr_packets,
    "ECE Flag Count": lambda r: r.bidirectional_ece_packets,
    "Down/Up Ratio":        lambda r: (r.dst2src_packets / r.src2dst_packets) if r.src2dst_packets > 0 else 0.0,
    "Average Packet Size":  lambda r: r.bidirectional_mean_ps,
    "Avg Fwd Segment Size": lambda r: r.src2dst_mean_ps,
    "Avg Bwd Segment Size": lambda r: r.dst2src_mean_ps,
    "Fwd Packets/s": lambda r: _rate(r.src2dst_packets, r.src2dst_duration_ms),
    "Bwd Packets/s": lambda r: _rate(r.dst2src_packets, r.dst2src_duration_ms),
    "Subflow Fwd Packets": lambda r: r.src2dst_packets,
    "Subflow Fwd Bytes":   lambda r: r.src2dst_bytes,
    "Subflow Bwd Packets": lambda r: r.dst2src_packets,
    "Subflow Bwd Bytes":   lambda r: r.dst2src_bytes,
    # Header lengths: 20 bytes for IP + 20 bytes for TCP base header × packet count
    "Fwd Header Length":   lambda r: r.src2dst_packets * 40,
    "Bwd Header Length":   lambda r: r.dst2src_packets * 40,
    "Fwd Header Length.1": lambda r: r.src2dst_packets * 40,
}


def _extract_feature(name: str, row) -> float:
    extractor = _EXTRACTORS.get(name)
    if extractor is None:
        return 0.0
    try:
        return float(extractor(row))
    except Exception:
        return 0.0


def parse_pcap(
    content: bytes,
    feature_names: list[str],
) -> tuple[list[dict[str, float]], list[dict]]:
    """
    Parse PCAP bytes and extract CIC-IDS2017-compatible flow features.

    Uses nfstream for per-flow statistical feature extraction. Features not
    derivable from nfstream are set to 0.0 (best-effort mapping).

    Returns:
        valid_rows    — list of {feature_name: float} dicts, one per TCP/UDP flow
        rejected_rows — list of {row: int, reason: str} for flows that errored
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as f:
            f.write(content)
            tmp_path = f.name

        try:
            from nfstream import NFStreamer
        except ImportError:
            return [], [{"row": -1, "reason": "nfstream not installed — cannot parse PCAP"}]

        try:
            df = NFStreamer(source=tmp_path, statistical_analysis=True).to_pandas()
        except Exception as exc:
            return [], [{"row": -1, "reason": f"PCAP parse error: {exc}"}]

        if df is None or df.empty:
            return [], []

        valid_rows: list[dict[str, float]] = []
        rejected_rows: list[dict] = []

        for i, row in df.iterrows():
            try:
                valid_rows.append({name: _extract_feature(name, row) for name in feature_names})
            except Exception as exc:
                rejected_rows.append({"row": int(i), "reason": str(exc)})

        return valid_rows, rejected_rows

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
