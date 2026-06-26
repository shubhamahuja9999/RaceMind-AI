"""Base callback interface for RaceMind AI training (Phase 2+).

Defines the lifecycle hooks a training loop will invoke. No reinforcement
learning logic lives here — these are reusable interfaces only. ``context`` is a
plain mutable dict so the training loop can pass arbitrary state (episode index,
step count, model handle, paths) without coupling callbacks to a concrete
trainer.
"""

from __future__ import annotations

from typing import Any, Iterable

Context = dict[str, Any]


class BaseCallback:
    """Lifecycle hooks invoked by a training loop. Override what you need."""

    def on_training_start(self, context: Context) -> None:
        """Called once before training begins."""

    def on_episode_start(self, context: Context) -> None:
        """Called at the start of each episode."""

    def on_step(self, context: Context) -> None:
        """Called after each environment step."""

    def on_episode_end(self, context: Context) -> None:
        """Called at the end of each episode."""

    def on_training_end(self, context: Context) -> None:
        """Called once after training finishes."""


class CallbackList(BaseCallback):
    """A composite callback that forwards every hook to its members in order."""

    def __init__(self, callbacks: Iterable[BaseCallback]) -> None:
        """Store the child callbacks.

        Args:
            callbacks: Callbacks to invoke, in iteration order.
        """
        self._callbacks: list[BaseCallback] = list(callbacks)

    def on_training_start(self, context: Context) -> None:
        for callback in self._callbacks:
            callback.on_training_start(context)

    def on_episode_start(self, context: Context) -> None:
        for callback in self._callbacks:
            callback.on_episode_start(context)

    def on_step(self, context: Context) -> None:
        for callback in self._callbacks:
            callback.on_step(context)

    def on_episode_end(self, context: Context) -> None:
        for callback in self._callbacks:
            callback.on_episode_end(context)

    def on_training_end(self, context: Context) -> None:
        for callback in self._callbacks:
            callback.on_training_end(context)
