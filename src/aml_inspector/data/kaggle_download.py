"""Optional helpers for bringing Kaggle IBM AML CSVs into data/raw.

Dataset: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

Typical workflow: install Kaggle CLI, set ~/.kaggle/kaggle.json, then:
  kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml -p data/raw --unzip
"""

from __future__ import annotations

from pathlib import Path


def raw_dir() -> Path:
    from aml_inspector.config import DATA_RAW

    return DATA_RAW


def print_manual_instructions() -> None:
    """Print how to obtain data without importing optional Kaggle SDK."""
    print(
        "Place unpacked CSV files under",
        raw_dir(),
        "or use: kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml -p data/raw --unzip",
    )
