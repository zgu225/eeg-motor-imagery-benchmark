"""Run the complete benchmark from the project root."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bci_project.benchmark import main

if __name__ == "__main__":
    main()
