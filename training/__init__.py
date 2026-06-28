"""Training infrastructure for RaceMind AI.

Submodules are intentionally not imported here to keep the package import-cycle
free (``evaluation`` imports :mod:`training.training_loop`, while
:mod:`training.trainer` imports ``evaluation``). Import the pieces you need
directly, e.g.::

    from training.trainer import Trainer, build_trainer
    from training.ppo_trainer import build_ppo_trainer
    from training.checkpoint_manager import CheckpointManager
    from training.training_loop import run_episode
"""

__all__: list[str] = []
