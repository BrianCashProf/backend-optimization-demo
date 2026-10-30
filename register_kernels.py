"""Register the workspace's standard and free-threaded Jupyter kernels.

Run after creating the environments: .venv/bin/python register_kernels.py
Use --local-only to keep registration inside .venv/share/jupyter/kernels.
"""

import argparse
import json
from pathlib import Path

from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parent


def register(local_only=False):
    standard = ROOT / ".venv/bin/python"
    free_threaded = ROOT / ".venv-free-threaded/bin/python"
    for executable in (standard, free_threaded):
        if not executable.is_file():
            raise FileNotFoundError(f"Create the virtual environment first: {executable}")
    profiles = [
        ("order-pricing-314", "Order Pricing - Python 3.14 (standard)", standard, []),
        ("order-pricing-314t-gil-on", "Order Pricing - Python 3.14t (GIL on)", free_threaded, ["-X", "gil=1"]),
        ("order-pricing-314t-gil-off", "Order Pricing - Python 3.14t (GIL off)", free_threaded, ["-X", "gil=0"]),
    ]
    manager = KernelSpecManager()
    for name, display_name, executable, flags in profiles:
        folder = ROOT / ".venv/share/jupyter/kernels" / name
        folder.mkdir(parents=True, exist_ok=True)
        spec = {
            "argv": [str(executable), *flags, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": display_name,
            "language": "python",
            "env": {"PRICING_FT_PYTHON": str(free_threaded)},
        }
        (folder / "kernel.json").write_text(json.dumps(spec, indent=2) + "\n")
        if not local_only:
            manager.install_kernel_spec(str(folder), kernel_name=name, user=True)
        print(f"{display_name}\n  {executable}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-only", action="store_true")
    register(parser.parse_args().local_only)
