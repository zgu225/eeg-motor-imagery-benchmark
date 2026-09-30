"""Command-line entry point for cross-session experiments."""

from pathlib import Path
import sys


# Add the source tree when this script is launched directly from VS Code.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bci_project.experiments.within_subject import main


if __name__ == "__main__":
    main()
