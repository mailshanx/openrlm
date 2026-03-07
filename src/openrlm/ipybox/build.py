"""Build the openrlm sandbox Docker image."""

import subprocess
import tempfile
from pathlib import Path


DOCKER_DIR = Path(__file__).parent / "docker"
KERNEL_PY = Path(__file__).parent / "kernel.py"


def build(tag: str, dependencies_file: Path | None = None):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        # Read extra dependencies
        deps_lines = ""
        if dependencies_file and dependencies_file.exists():
            deps_text = dependencies_file.read_text().strip()
            if deps_text:
                deps_lines = "\n" + deps_text

        # Template the pyproject.toml
        template = (DOCKER_DIR / "pyproject.toml").read_text()
        (tmp / "pyproject.toml").write_text(template.replace("{dependencies}", deps_lines))

        # Copy static files
        for name in ("Dockerfile", ".python-version"):
            (tmp / name).write_text((DOCKER_DIR / name).read_text())

        # Copy kernel server
        (tmp / "kernel.py").write_text(KERNEL_PY.read_text())

        cmd = ["docker", "build", "-f", str(tmp / "Dockerfile"), "-t", tag, str(tmp)]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                print(output.strip())

        if process.returncode != 0:
            raise SystemExit(f"Docker build failed with exit code {process.returncode}")
