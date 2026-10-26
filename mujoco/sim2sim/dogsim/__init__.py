"""Modular MuJoCo sim2sim runtime for the HighTorque MiniDog / Go2W tasks.

The package is intentionally split into replaceable bricks:

- ``robots``      robot profiles (joint order, gains, stand pose)
- ``obstacles``   independent obstacle modules with a small build contract
- ``policies``    checkpoint loaders and inference-only actor implementations
- ``observation``  observation terms, history and the height scanner
- ``simulation``  one stepping runtime, independent of terrain or policy type
- ``evaluators``  one success criterion per obstacle family
- ``suite``       matrix runner and report writers

Nothing in ``simulation`` knows a concrete obstacle, and nothing in an
obstacle knows a policy.  Adding a new obstacle means adding one module file
and one evaluator file; the CLI and suite pick them up through the registry.
"""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "2.0.0"
