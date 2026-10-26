from __future__ import annotations


class OnPolicyRunner:
    """Placeholder for the native RSL-RL runner port.

    The active MJLab training entry currently delegates to SB3 PPO.  This class
    keeps the directory compatible with the usual legged_gym/rsl_rl layout while
    leaving a clear target for a later native RSL-RL implementation.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Native RSL-RL runner is not wired yet; use legged_gym/scripts/train.py.")

