from __future__ import annotations

import importlib
import os
import sys
from importlib import metadata
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_DIRS = {
    "JUPYTER_CONFIG_DIR": PROJECT_ROOT / ".jupyter",
    "JUPYTER_DATA_DIR": PROJECT_ROOT / ".jupyter" / "data",
    "JUPYTER_RUNTIME_DIR": PROJECT_ROOT / ".jupyter" / "runtime",
    "IPYTHONDIR": PROJECT_ROOT / ".ipython",
    "MPLCONFIGDIR": PROJECT_ROOT / ".cache" / "matplotlib",
    "PIP_CACHE_DIR": PROJECT_ROOT / ".cache" / "pip",
    "PYTHONPYCACHEPREFIX": PROJECT_ROOT / ".cache" / "pycache",
}
CHECKS = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("plotly", "plotly"),
    ("scikit-learn", "sklearn"),
    ("keras", "keras"),
    ("tensorflow", "tensorflow"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("torchaudio", "torchaudio"),
    ("gymnasium", "gymnasium"),
    ("stable-baselines3", "stable_baselines3"),
    ("jupyterlab", "jupyterlab"),
    ("ipykernel", "ipykernel"),
    ("ipywidgets", "ipywidgets"),
    ("anywidget", "anywidget"),
]
OPTIONAL_DISTRIBUTIONS = {
    "keras",
    "tensorflow",
    "torchvision",
    "torchaudio",
    "gymnasium",
    "stable-baselines3",
    "ipywidgets",
    "anywidget",
}


def ensure_local_runtime_dirs() -> None:
    for value in LOCAL_DIRS.values():
        value.mkdir(parents=True, exist_ok=True)

    for key, value in LOCAL_DIRS.items():
        os.environ.setdefault(key, str(value))

    src_dir = PROJECT_ROOT / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))


def installed_version(distribution_name: str) -> str:
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return "missing"


def import_check(distribution_name: str, module_name: str) -> None:
    version = installed_version(distribution_name)
    try:
        module = importlib.import_module(module_name)
        runtime_version = getattr(module, "__version__", version)
        print(f"[OK] {distribution_name:<18} installed={version} runtime={runtime_version}")
    except Exception as exc:
        level = "WARN" if distribution_name in OPTIONAL_DISTRIBUTIONS else "ERRO"
        print(f"[{level}] {distribution_name:<18} installed={version} -> {exc}")


def print_header() -> None:
    print(f"Project: {PROJECT_ROOT}")
    print(f"Python:  {sys.version.split()[0]}")
    print(f"Exec:    {sys.executable}")
    print("Local paths:")
    for key, value in LOCAL_DIRS.items():
        print(f"  - {key}={value}")
    print()


def check_torch() -> None:
    try:
        import torch

        print()
        print(f"[Torch] version={torch.__version__} cuda_build={torch.version.cuda}")
        print(f"[Torch] cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[Torch] device={torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"[Torch] error -> {exc}")


def check_tensorflow() -> None:
    try:
        import tensorflow as tf

        print()
        print(f"[TensorFlow] version={tf.__version__}")
        print(f"[TensorFlow] built_with_cuda={tf.test.is_built_with_cuda()}")
        print(f"[TensorFlow] physical_gpus={tf.config.list_physical_devices('GPU')}")
    except Exception as exc:
        print(f"[TensorFlow] error -> {exc}")


def main() -> None:
    ensure_local_runtime_dirs()
    print_header()
    for distribution_name, module_name in CHECKS:
        import_check(distribution_name, module_name)
    check_torch()
    check_tensorflow()


if __name__ == "__main__":
    main()
