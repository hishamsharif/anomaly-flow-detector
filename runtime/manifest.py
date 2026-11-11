from __future__ import annotations

from runtime.probe import HostEnvironment, probe_environment

_env: HostEnvironment | None = None


def resolve_host_environment() -> HostEnvironment:
    """Probe the host and cache the result for the process lifetime."""
    global _env
    _env = probe_environment()
    return _env


def get_host_environment() -> HostEnvironment:
    """Return the cached environment, probing on first call if needed."""
    if _env is None:
        return resolve_host_environment()
    return _env
