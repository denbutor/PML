"""Безпечне завантаження skops-артефактів: довіряємо лише типам з білого списку (README, «Довірені типи skops»)."""
from __future__ import annotations

from pathlib import Path

TRUSTED_PREFIXES = ("src.", "sklearn.", "numpy.", "builtins.", "collections.", "scipy.", "pandas.", "functools.")


def load_skops(path: str | Path):
    import skops.io as sio

    untrusted = sio.get_untrusted_types(file=str(path))
    foreign = [t for t in untrusted if not t.startswith(TRUSTED_PREFIXES)]
    if foreign:
        raise ValueError(f"{path}: недовірені типи поза білим списком: {foreign}")
    return sio.load(str(path), trusted=untrusted)
