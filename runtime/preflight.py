from dataclasses import asdict, dataclass

from runtime.probe import HostEnvironment


@dataclass
class CheckResult:
    level: str  # "ok" | "warn" | "blocker"
    message: str
    detail: str | None = None


def _check_cuda(env: HostEnvironment) -> CheckResult:
    if env.cuda_available:
        return CheckResult("ok", "CUDA available", f"CUDA {env.cuda_version}")
    return CheckResult(
        "warn",
        "CUDA not available",
        "Training will fall back to CPU — slower but fully functional.",
    )


def _check_docker(env: HostEnvironment) -> CheckResult:
    if env.docker_available:
        return CheckResult("ok", "Docker available", f"Docker {env.docker_version}")
    return CheckResult(
        "warn",
        "Docker not found",
        "Required for `make run`. Not needed for local dev (`make run-api` + `make run-ui`).",
    )


def _check_images(env: HostEnvironment) -> CheckResult:
    if not env.docker_available:
        return CheckResult("warn", "Docker unavailable — image check skipped")
    expected_prefixes = ["anomaly-flow-detector-api", "anomaly-flow-detector-ui"]
    missing = [p for p in expected_prefixes if not any(p in img for img in env.docker_images)]
    if not missing:
        return CheckResult("ok", "Required Docker images present")
    return CheckResult(
        "warn",
        "Docker images not built yet",
        "Run `make run` once to build them.",
    )


def build_preflight_report(env: HostEnvironment) -> dict:
    return {
        "cuda": asdict(_check_cuda(env)),
        "docker": asdict(_check_docker(env)),
        "images": asdict(_check_images(env)),
    }
