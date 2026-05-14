"""Download IBM AML transaction CSVs and laundering pattern text files from Kaggle.

Dataset: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

Each requested file is downloaded with a separate ``kaggle datasets download`` invocation
so that skipping an up-to-date local file (e.g. ``HI-Medium_Trans.csv``) does not block
other files (e.g. ``HI-Small_Trans.csv``) from being fetched.

Requires the Kaggle CLI (``pip install kaggle``) and credentials at ``~/.kaggle/kaggle.json``.

Typical workflow::

  python -m aml_inspector.data.kaggle_download
  python -m aml_inspector.data.kaggle_download --output-dir /path/to/raw
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DATASET_SLUG = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
DEFAULT_TRANS_FILES = ("HI-Small_Trans.csv", "HI-Medium_Trans.csv")
# Laundering pattern descriptions (HI/LI × Small/Medium/Large).
DEFAULT_PATTERN_FILES = (
    "HI-Small_Patterns.txt",
    "HI-Medium_Patterns.txt",
    "HI-Large_Patterns.txt",
    "LI-Small_Patterns.txt",
    "LI-Medium_Patterns.txt",
    "LI-Large_Patterns.txt",
)
DEFAULT_FILES = DEFAULT_TRANS_FILES + DEFAULT_PATTERN_FILES


def raw_dir() -> Path:
    from aml_inspector.config import DATA_RAW

    return DATA_RAW


def print_manual_instructions() -> None:
    """Print how to obtain data without the Kaggle CLI."""
    print(
        "Place unpacked CSV files under",
        raw_dir(),
        "or use: kaggle datasets download -d",
        DATASET_SLUG,
        "-p data/raw --unzip",
    )


def _which_kaggle() -> str | None:
    exe = shutil.which("kaggle")
    if exe:
        return exe
    # Venv layout: Scripts/kaggle next to python.exe (Windows) or bin/kaggle (Unix).
    bindir = Path(sys.executable).resolve().parent
    name = "kaggle.exe" if sys.platform == "win32" else "kaggle"
    candidate = bindir / name
    if candidate.is_file():
        return str(candidate)
    return None


def _download_one_file(
    exe: str,
    output_dir: Path,
    name: str,
    *,
    unzip: bool,
    force: bool,
) -> None:
    """Run Kaggle CLI for a single ``-f`` so per-file skip logic cannot strand siblings."""
    cmd: list[str] = [
        exe,
        "datasets",
        "download",
        "-d",
        DATASET_SLUG,
        "-p",
        str(output_dir),
        "-f",
        name,
    ]
    if unzip:
        cmd.append("--unzip")
    if force:
        cmd.append("--force")

    print("Running:", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(
            "ERROR: kaggle datasets download failed for",
            repr(name),
            "(exit code",
            result.returncode,
            "). Check credentials and dataset slug.",
            file=sys.stderr,
        )
        sys.exit(result.returncode)

    # If Kaggle returned a zip without auto-unzip, extract then remove the zip.
    if not unzip:
        for z in output_dir.glob("*.zip"):
            with zipfile.ZipFile(z, "r") as zf:
                zf.extractall(output_dir)
            z.unlink(missing_ok=True)

    if not (output_dir / name).is_file():
        print(
            "ERROR: Expected file not found after download:",
            name,
            "in",
            output_dir,
            file=sys.stderr,
        )
        sys.exit(1)
    print("OK:", output_dir / name, flush=True)


def _run_kaggle_download(
    output_dir: Path,
    files: tuple[str, ...],
    unzip: bool,
    *,
    force: bool,
) -> None:
    exe = _which_kaggle()
    if exe is None:
        print(
            "ERROR: `kaggle` CLI not found. Install with: pip install kaggle\n"
            "Then add API credentials to ~/.kaggle/kaggle.json (see Kaggle account settings).",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    for name in files:
        _download_one_file(exe, output_dir, name, unzip=unzip, force=force)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            f"Download IBM AML transaction CSVs and laundering pattern files from Kaggle "
            f"({DATASET_SLUG})."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Directory for CSVs (default: project data/raw = {raw_dir()})",
    )
    parser.add_argument(
        "--no-unzip",
        action="store_true",
        help="Do not pass --unzip to kaggle; extract any .zip manually in script.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pass --force to kaggle (re-download even if local copy looks up to date).",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=list(DEFAULT_FILES),
        help=(
            "Filenames to fetch (default: HI Small/Medium transaction CSVs plus all "
            "*_Patterns.txt laundering pattern files)"
        ),
    )
    parser.add_argument(
        "--instructions-only",
        action="store_true",
        help="Print manual download instructions and exit.",
    )
    args = parser.parse_args(argv)

    if args.instructions_only:
        print_manual_instructions()
        return

    out = args.output_dir if args.output_dir is not None else raw_dir()
    files = tuple(args.files)
    _run_kaggle_download(out, files, unzip=not args.no_unzip, force=args.force)


if __name__ == "__main__":
    main()
