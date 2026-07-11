"""Off-track penalty component."""

from __future__ import annotations

import numpy as np

from reward.base_reward import BaseRewardComponent, RewardContext

# A small patch of the observation just below the car; grass (off-track) is
# green-dominant there, road is grey.
_PATCH_ROWS = slice(66, 78)
_PATCH_COLS = slice(42, 54)
_GREEN_MARGIN = 20.0


class OffTrackPenalty(BaseRewardComponent):
    """Penalises leaving the track, detected by a grass (green) heuristic.

    ``CarRacing-v3`` does not expose an off-track flag, so this reads
    ``context.info['off_track']`` when available and otherwise falls back to a
    simple heuristic on the RGB observation: if the patch below the car is
    strongly green-dominant (grass), the car is off-track. Returns ``-1.0`` when
    off-track, ``0.0`` otherwise (weight applies the penalty scale).
    """

    name = "offtrack"

    def compute_reward(self, context: RewardContext) -> float:
        """Return ``-1.0`` if off-track, else ``0.0``."""
        flag = context.info.get("off_track")
        if flag is not None:
            return -1.0 if bool(flag) else 0.0
        return -1.0 if self._is_on_grass(context.observation) else 0.0

    @staticmethod
    def _is_on_grass(observation: np.ndarray) -> bool:
        """Heuristic: is the patch below the car predominantly green (grass)?"""
        if observation.ndim != 3 or observation.shape[2] < 3:
            return False
        patch = observation[_PATCH_ROWS, _PATCH_COLS, :].astype(np.float32)
        red, green, blue = patch[..., 0].mean(), patch[..., 1].mean(), patch[..., 2].mean()
        return green > red + _GREEN_MARGIN and green > blue + _GREEN_MARGIN
