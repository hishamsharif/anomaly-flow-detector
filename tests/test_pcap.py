"""
PCAP adapter tests.

Skipped automatically when nfstream is not installed.
The fixture PCAP is built from raw struct bytes — no scapy dependency.
"""
from __future__ import annotations

import struct
from unittest.mock import MagicMock, patch

import pytest

# ── Minimal PCAP builder (no scapy) ──────────────────────────────────────────

_PCAP_GLOBAL_HDR = struct.pack(
    "<IHHiIII",
    0xA1B2C3D4,  # magic (little-endian)
    2, 4,        # version major, minor
    0,           # timezone
    0,           # sig figs
    65535,       # snaplen
    1,           # link type: Ethernet
)


def _eth_ip_tcp(
    src_ip: tuple[int, ...],
    dst_ip: tuple[int, ...],
    src_port: int,
    dst_port: int,
    seq: int = 1000,
    ack: int = 0,
    flags: int = 0x002,  # SYN
) -> bytes:
    """Build a minimal Ethernet + IP + TCP frame (no data, no valid checksums)."""
    # Ethernet header (14 bytes)
    eth = bytes([0, 0, 0, 0, 0, 2])   # dst mac
    eth += bytes([0, 0, 0, 0, 0, 1])  # src mac
    eth += struct.pack("!H", 0x0800)  # IPv4

    ip_len = 40  # 20 IP + 20 TCP
    # IPv4 header (20 bytes): version/IHL, DSCP, total_len, id, flags+frag,
    #                          TTL, proto, checksum, src, dst
    ip = struct.pack(
        "!BBHHHBBH4s4s",
        0x45, 0, ip_len,
        1, 0x4000,
        64, 6, 0,
        bytes(src_ip), bytes(dst_ip),
    )

    # TCP header (20 bytes): sport, dport, seq, ack, data_off, flags, win, chk, urg
    tcp = struct.pack(
        "!HHLLBBHHH",
        src_port, dst_port,
        seq, ack,
        0x50, flags,
        65535, 0, 0,
    )

    return eth + ip + tcp


def _pcap_record(payload: bytes, ts_sec: int = 1000000, ts_usec: int = 0) -> bytes:
    return struct.pack("<IIII", ts_sec, ts_usec, len(payload), len(payload)) + payload


def _build_pcap_bytes(n_flows: int = 2, pkts_per_flow: int = 3) -> bytes:
    """
    Build a PCAP with `n_flows` distinct TCP flows, each with `pkts_per_flow`
    packets (SYN → ACK → FIN sequence).
    """
    records = b""
    for flow_idx in range(n_flows):
        src_port = 50000 + flow_idx
        for pkt_idx in range(pkts_per_flow):
            # Alternate between SYN / ACK / FIN
            flags = [0x002, 0x010, 0x011][pkt_idx % 3]
            frame = _eth_ip_tcp(
                src_ip=(192, 168, 1, 1),
                dst_ip=(10, 0, 0, flow_idx + 1),
                src_port=src_port,
                dst_port=80,
                seq=1000 + pkt_idx,
                ack=2000 + pkt_idx if pkt_idx > 0 else 0,
                flags=flags,
            )
            ts = 1_000_000 + flow_idx * 1000 + pkt_idx * 100
            records += _pcap_record(frame, ts_sec=ts)

    return _PCAP_GLOBAL_HDR + records


def _build_empty_pcap() -> bytes:
    return _PCAP_GLOBAL_HDR


# ── Tests that run without nfstream ──────────────────────────────────────────

class TestPcapAdapterNoNfstream:
    """Unit-test the adapter's import-guard path without needing nfstream."""

    def test_returns_rejection_when_nfstream_missing(self):
        with patch.dict("sys.modules", {"nfstream": None}):
            from inference.adapters import pcap_adapter
            import importlib
            importlib.reload(pcap_adapter)

            valid, rejected = pcap_adapter.parse_pcap(b"dummy", ["Total Fwd Packets"])

        assert valid == []
        assert len(rejected) == 1
        assert "nfstream" in rejected[0]["reason"].lower()

    def test_returns_rejection_on_parse_error(self, tmp_path):
        """NFStreamer raises → returned as a single rejection row."""
        mock_nfstream = MagicMock()
        mock_nfstream.NFStreamer.side_effect = Exception("bad pcap")

        with patch.dict("sys.modules", {"nfstream": mock_nfstream}):
            from inference.adapters import pcap_adapter
            import importlib
            importlib.reload(pcap_adapter)

            valid, rejected = pcap_adapter.parse_pcap(b"not-a-pcap", ["Total Fwd Packets"])

        assert valid == []
        assert len(rejected) == 1
        assert "bad pcap" in rejected[0]["reason"]

    def test_empty_dataframe_returns_empty_lists(self):
        """NFStreamer returns empty DataFrame → ([], [])."""
        import pandas as pd

        mock_nfstream = MagicMock()
        mock_streamer = MagicMock()
        mock_streamer.to_pandas.return_value = pd.DataFrame()
        mock_nfstream.NFStreamer.return_value = mock_streamer

        with patch.dict("sys.modules", {"nfstream": mock_nfstream}):
            from inference.adapters import pcap_adapter
            import importlib
            importlib.reload(pcap_adapter)

            valid, rejected = pcap_adapter.parse_pcap(_build_empty_pcap(), [])

        assert valid == []
        assert rejected == []

    def test_unmapped_feature_defaults_to_zero(self):
        """Features absent from _EXTRACTORS → 0.0, not KeyError."""
        import pandas as pd

        # One synthetic flow row
        row = pd.Series({"bidirectional_duration_ms": 50.0, "src2dst_packets": 3})
        df = pd.DataFrame([row])

        mock_nfstream = MagicMock()
        mock_streamer = MagicMock()
        mock_streamer.to_pandas.return_value = df
        mock_nfstream.NFStreamer.return_value = mock_streamer

        with patch.dict("sys.modules", {"nfstream": mock_nfstream}):
            from inference.adapters import pcap_adapter
            import importlib
            importlib.reload(pcap_adapter)

            valid, rejected = pcap_adapter.parse_pcap(b"x", ["DOES_NOT_EXIST_XYZ"])

        assert len(valid) == 1
        assert valid[0]["DOES_NOT_EXIST_XYZ"] == 0.0
        assert rejected == []

    def test_mapped_feature_extracted_correctly(self):
        """Flow Duration extractor: bidirectional_duration_ms * 1000."""
        import pandas as pd

        row = pd.Series({
            "bidirectional_duration_ms": 100.0,
            "src2dst_packets": 3,
            "dst2src_packets": 2,
        })
        df = pd.DataFrame([row])

        mock_nfstream = MagicMock()
        mock_streamer = MagicMock()
        mock_streamer.to_pandas.return_value = df
        mock_nfstream.NFStreamer.return_value = mock_streamer

        with patch.dict("sys.modules", {"nfstream": mock_nfstream}):
            from inference.adapters import pcap_adapter
            import importlib
            importlib.reload(pcap_adapter)

            valid, _ = pcap_adapter.parse_pcap(b"x", ["Flow Duration", "Total Fwd Packets"])

        assert valid[0]["Flow Duration"] == pytest.approx(100_000.0)
        assert valid[0]["Total Fwd Packets"] == pytest.approx(3.0)

    def test_feature_dict_keys_match_requested_names(self):
        """Every row must have exactly the requested feature names as keys."""
        import pandas as pd

        row = pd.Series({"bidirectional_duration_ms": 1.0, "src2dst_packets": 1})
        df = pd.DataFrame([row, row])

        mock_nfstream = MagicMock()
        mock_streamer = MagicMock()
        mock_streamer.to_pandas.return_value = df
        mock_nfstream.NFStreamer.return_value = mock_streamer

        feature_names = ["Flow Duration", "Total Fwd Packets", "UNKNOWN_FEAT"]

        with patch.dict("sys.modules", {"nfstream": mock_nfstream}):
            from inference.adapters import pcap_adapter
            import importlib
            importlib.reload(pcap_adapter)

            valid, _ = pcap_adapter.parse_pcap(b"x", feature_names)

        assert len(valid) == 2
        for row_dict in valid:
            assert set(row_dict.keys()) == set(feature_names)


# ── Integration tests (require nfstream) ─────────────────────────────────────

nfstream = pytest.importorskip("nfstream", reason="nfstream not installed")


class TestPcapAdapterIntegration:
    """Live nfstream tests — only run when nfstream is installed."""

    def test_valid_pcap_returns_at_least_one_flow(self):
        from inference.adapters.pcap_adapter import parse_pcap

        pcap = _build_pcap_bytes(n_flows=2, pkts_per_flow=3)
        feature_names = ["Total Fwd Packets", "Total Backward Packets", "Flow Duration"]
        valid, rejected = parse_pcap(pcap, feature_names)

        assert len(valid) >= 1

    def test_all_returned_values_are_floats(self):
        from inference.adapters.pcap_adapter import parse_pcap

        pcap = _build_pcap_bytes(n_flows=1, pkts_per_flow=3)
        feature_names = [
            "Total Fwd Packets", "Total Backward Packets", "Flow Duration",
            "SYN Flag Count", "ACK Flag Count", "FIN Flag Count",
        ]
        valid, _ = parse_pcap(pcap, feature_names)

        for row in valid:
            assert set(row.keys()) == set(feature_names)
            for v in row.values():
                assert isinstance(v, float), f"Expected float, got {type(v)} for value {v!r}"

    def test_empty_pcap_returns_empty_lists(self):
        from inference.adapters.pcap_adapter import parse_pcap

        valid, rejected = parse_pcap(_build_empty_pcap(), ["Total Fwd Packets"])
        assert valid == []
        assert rejected == []
