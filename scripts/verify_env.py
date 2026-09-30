"""Verify that the BCI project environment is ready."""

from importlib import import_module
from pathlib import Path
import sys


# Add the source tree when this script is launched directly from VS Code.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


PACKAGES = [
    "braindecode",
    "matplotlib",
    "mne",
    "moabb",
    "numpy",
    "pandas",
    "pyriemann",
    "sklearn",
    "scipy",
    "seaborn",
    "torch",
    "tqdm",
]


def main() -> None:
    # Import each dependency to verify that the environment is complete.
    for package in PACKAGES:
        module = import_module(package)
        version = getattr(module, "__version__", "unknown")
        print(f"{package}: {version}")

    # Check that the principal project modules are present.
    required_paths = [
        PROJECT_ROOT / "configs/bnci2014_001.yaml",
        PROJECT_ROOT / "src/bci_project/data/load_moabb.py",
        PROJECT_ROOT / "src/bci_project/models/lawhern_eegnet.py",
        PROJECT_ROOT / "src/bci_project/models/deep_models.py",
        PROJECT_ROOT / "src/bci_project/experiments/within_subject.py",
        PROJECT_ROOT / "src/bci_project/experiments/cross_subject.py",
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing project file: {path}")

    print("BCI environment is ready.")


if __name__ == "__main__":
    main()
