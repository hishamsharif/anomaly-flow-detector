import shutil
import subprocess
from dataclasses import dataclass, field


@dataclass
class HostEnvironment:
    cuda_available: bool = False
    cuda_version: str | None = None
    docker_available: bool = False
    docker_version: str | None = None
    docker_images: list[str] = field(default_factory=list)


def probe_cuda() -> tuple[bool, str | None]:
    try:
        import torch  # noqa: PLC0415
        available = torch.cuda.is_available()
        version = torch.version.cuda if available else None
        return available, version
    except ImportError:
        return False, None


def probe_docker() -> tuple[bool, str | None]:
    if not shutil.which("docker"):
        return False, None
    try:
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return True, result.stdout.strip()
        return False, None
    except Exception:
        return False, None


def probe_images() -> list[str]:
    try:
        result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return [line for line in result.stdout.strip().splitlines() if line and line != "<none>:<none>"]
        return []
    except Exception:
        return []


def probe_environment() -> HostEnvironment:
    cuda_ok, cuda_ver = probe_cuda()
    docker_ok, docker_ver = probe_docker()
    images = probe_images() if docker_ok else []
    return HostEnvironment(
        cuda_available=cuda_ok,
        cuda_version=cuda_ver,
        docker_available=docker_ok,
        docker_version=docker_ver,
        docker_images=images,
    )
