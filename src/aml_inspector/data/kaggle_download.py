"""Download IBM AML Small HI and Medium HI CSVs from Kaggle into data/raw.

Dataset: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

Requires the Kaggle CLI (`pip install kaggle`) and credentials at ``~/.kaggle/kaggle.json``.

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
DEFAULT_FILES = ("HI-Small_Trans.csv", "HI-Medium_Trans.csv")


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
    return shutil.which("kaggle")


def _run_kaggle_download(
    output_dir: Path,
    files: tuple[str, ...],
    unzip: bool,
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

    cmd: list[str] = [
        exe,
        "datasets",
        "download",
        "-d",
        DATASET_SLUG,
        "-p",
        str(output_dir),
    ]
    for name in files:
        cmd.extend(["-f", name])
    if unzip:
        cmd.append("--unzip")

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(
            "ERROR: kaggle datasets download failed (exit code",
            result.returncode,
            "). Check credentials and dataset slug.",
            file=sys.stderr,
        )
        sys.exit(result.returncode)

    # If Kaggle returned a single zip without auto-unzip, extract CSVs.
    if not unzip:
        for z in output_dir.glob("*.zip"):
            with zipfile.ZipFile(z, "r") as zf:
                zf.extractall(output_dir)
            z.unlink(missing_ok=True)

    missing = [f for f in files if not (output_dir / f).is_file()]
    if missing:
        print(
            "ERROR: Expected files not found after download:",
            ", ".join(missing),
            "in",
            output_dir,
            file=sys.stderr,
        )
        sys.exit(1)

    for f in files:
        print("OK:", output_dir / f)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=f"Download IBM AML HI CSVs from Kaggle ({DATASET_SLUG}).",
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
        "--files",
        nargs="*",
        default=list(DEFAULT_FILES),
        help=f"CSV filenames to fetch (default: {' '.join(DEFAULT_FILES)})",
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
    _run_kaggle_download(out, files, unzip=not args.no_unzip)


if __name__ == "__main__":
    main()
