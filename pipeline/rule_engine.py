from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuleEngine:
    """
    Stage 1 deterministic rules — always evaluated first; always take precedence
    over ML output (§3.3, SPEC §5.2).

    Rules operate on the raw feature dict for a single flow.  Add entries to
    `port_verdict_rules` or `allowlisted_ports` at runtime without retraining.

    CIC-IDS2017 flows carry statistical features (no raw IPs), so rules here
    target `Destination Port` and flow-statistic thresholds typical of attack
    signatures in the dataset.
    """

    # Destination port → forced verdict.
    # Ports are not conclusive by themselves in CIC-IDS2017; kept empty by
    # default so nearly all flows pass to Stage 2.  Extend via `with_defaults()`.
    port_verdict_rules: dict[int, str] = field(default_factory=dict)

    # Destination ports that are trusted management services (→ SAFE).
    allowlisted_ports: frozenset[int] = field(
        default_factory=lambda: frozenset()
    )

    def check(self, flow: dict[str, Any]) -> str | None:
        """
        Evaluate all configured rules against one flow.

        Returns:
            'KNOWN_ATTACK' — deterministic attack signature matched
            'SAFE'         — flow matches an allowlisted / trusted source
            None           — ambiguous; pass to Stage 2 (XGBoost classifier)
        """
        dst_port = flow.get("Destination Port")
        if dst_port is not None:
            try:
                port = int(dst_port)
            except (ValueError, TypeError):
                port = None

            if port is not None:
                if port in self.allowlisted_ports:
                    return "SAFE"
                forced = self.port_verdict_rules.get(port)
                if forced is not None:
                    return forced

        return None  # ambiguous — Stage 2 handles it

    @classmethod
    def with_defaults(cls) -> "RuleEngine":
        """
        Production rule set.  By design, the CIC-IDS2017 feature space (flow
        statistics, no IP columns) means very few flows can be deterministically
        classified at Stage 1.  Rules here are illustrative; extend with
        site-specific signatures without retraining.
        """
        return cls(
            port_verdict_rules={
                21:   "KNOWN_ATTACK",   # FTP brute-force (FTP-Patator in CIC-IDS2017)
                22:   "KNOWN_ATTACK",   # SSH brute-force (SSH-Patator in CIC-IDS2017)
                4444: "KNOWN_ATTACK",   # Metasploit reverse shell (illustrative)
            },
            allowlisted_ports=frozenset({
                123,   # NTP — trusted clock-sync traffic
                161,   # SNMP — network management
            }),
        )
