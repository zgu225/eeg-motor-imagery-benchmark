"""Build a portable source/results archive using an explicit allowlist."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "dist"
    destination.mkdir(exist_ok=True)
    archive = destination / "eeg-motor-imagery-portfolio.zip"
    paths = []
    for name in ("README.md", "RESULT_REPORT.md", "PROJECT_PLAN.md", "requirements.txt",
                 "requirements-lock.txt", ".gitignore"):
        paths.append(ROOT / name)
    for folder, suffixes in (("src", {".py"}), ("scripts", {".py"}),
                             ("tests", {".py"}), ("configs", {".yaml"}),
                             (".vscode", {".json"}), (".github", {".yml"})):
        paths.extend(p for p in (ROOT / folder).rglob("*") if p.is_file() and p.suffix in suffixes)
    results = ROOT / "reports/benchmark_v2"
    paths.extend(p for p in results.glob("*") if p.is_file() and p.suffix in {".csv", ".json", ".png", ".md"})
    # Per-trial evidence is small enough to share; raw signals and model binaries are not included.
    for name in ("predictions.csv", "metrics.json", "history.json", "confusion.json", "split.npz", "input.json"):
        paths.extend(results.rglob(name))
    with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
        for path in sorted(set(paths)):
            if path.is_file():
                bundle.write(path, str(path.relative_to(ROOT)))
    with ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError("Release archive failed CRC validation")
        print(f"Release verified: {archive} ({len(bundle.namelist())} files)")


if __name__ == "__main__":
    main()
