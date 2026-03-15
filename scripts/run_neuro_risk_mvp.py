#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _set_local_runtime_env() -> None:
    local_dirs = {
        "JUPYTER_CONFIG_DIR": PROJECT_ROOT / ".jupyter",
        "JUPYTER_DATA_DIR": PROJECT_ROOT / ".jupyter" / "data",
        "JUPYTER_RUNTIME_DIR": PROJECT_ROOT / ".jupyter" / "runtime",
        "IPYTHONDIR": PROJECT_ROOT / ".ipython",
        "MPLCONFIGDIR": PROJECT_ROOT / ".cache" / "matplotlib",
        "PIP_CACHE_DIR": PROJECT_ROOT / ".cache" / "pip",
        "PYTHONPYCACHEPREFIX": PROJECT_ROOT / ".cache" / "pycache",
    }
    for key, value in local_dirs.items():
        value.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(key, str(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the neuro-inspired risk classification MVP.")
    parser.add_argument("--epochs", type=int, default=None, help="Override the lightweight training epoch budget.")
    parser.add_argument("--mc-samples", type=int, default=None, help="Override the number of MC dropout passes.")
    parser.add_argument("--device", type=str, default=None, help="Device override, for example cpu or cuda.")
    return parser.parse_args()


def main() -> None:
    _set_local_runtime_env()
    args = parse_args()

    from neuro_risk.config import ProjectConfig
    from neuro_risk.workflow import run_research_prototype

    config = ProjectConfig()
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.mc_samples is not None:
        config.training.mc_samples = args.mc_samples
    if args.device is not None:
        config.training.device = args.device

    artifacts = run_research_prototype(config)
    print(f"checkpoint={artifacts.checkpoint_path}")
    print(f"report={artifacts.report_path}")
    print(f"figures={artifacts.figures_dir}")
    print(f"interactive_plot={artifacts.interactive_plot_path}")
    print(f"jsviz_payload={artifacts.jsviz_payload_path}")


if __name__ == "__main__":
    main()
