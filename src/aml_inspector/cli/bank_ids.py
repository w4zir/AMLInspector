"""Parse bank id CLI tokens (integers and inclusive ranges)."""

from __future__ import annotations

import argparse


def parse_bank_id_token(token: str) -> list[int]:
    """Parse one token as a bank id or inclusive range (e.g. ``30-33`` -> 30..33)."""
    text = str(token).strip()
    if not text:
        raise ValueError("empty bank id token")

    if "-" in text:
        start_s, end_s = text.split("-", 1)
        if not start_s.isdigit() or not end_s.isdigit():
            raise ValueError(f"invalid bank id token: {token!r}")
        start, end = int(start_s), int(end_s)
        if start > end:
            raise ValueError(f"invalid bank id range: {token!r} (start > end)")
        return list(range(start, end + 1))

    try:
        return [int(text)]
    except ValueError as exc:
        raise ValueError(f"invalid bank id token: {token!r}") from exc


def parse_bank_id_list(tokens: list[str] | None) -> list[int] | None:
    """Expand CLI bank id tokens into a flat list of integers."""
    if tokens is None:
        return None
    expanded: list[int] = []
    for token in tokens:
        expanded.extend(parse_bank_id_token(token))
    return expanded


class BankIdListAction(argparse.Action):
    """Flatten bank id / range tokens from ``nargs='+'`` into ``list[int]``."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: list[str] | None,
        option_string: str | None = None,
    ) -> None:
        if values is None:
            setattr(namespace, self.dest, None)
            return
        expanded: list[int] = []
        for token in values:
            expanded.extend(parse_bank_id_token(token))
        setattr(namespace, self.dest, expanded)


BANK_ID_LIST_HELP_SUFFIX = (
    "Each value is a bank id or inclusive range (e.g. 30-33). "
    "Multiple values expand to a flat list."
)
