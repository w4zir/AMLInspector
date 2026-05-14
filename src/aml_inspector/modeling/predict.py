"""Batch scoring entrypoint (extend with loaded MLflow model or sklearn pipeline)."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch predict placeholder.")
    parser.add_argument("--input-path", required=True, help="Input table path (parquet/csv)")
    parser.add_argument("--output-path", required=True, help="Output predictions path")
    args = parser.parse_args()
    raise SystemExit(
        f"Implement scoring for {args.input_path} -> {args.output_path} after model training is defined."
    )


if __name__ == "__main__":
    main()
