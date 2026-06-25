"""RaceMind AI simulator package.

Phase 1 building blocks: a thin environment wrapper, manual keyboard driving,
per-frame telemetry, episode recording and standalone replay.

Submodules are intentionally **not** imported here. Each has different
dependencies (only ``env``/``manual_drive`` require Gymnasium), so importing
them lazily keeps lightweight components — notably ``replay`` and ``utils`` —
usable without the full simulator stack installed.
"""

__all__: list[str] = []
