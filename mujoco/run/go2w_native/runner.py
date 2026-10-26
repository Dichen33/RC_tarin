"""Compatibility exports for the native Go2W runner.

Go2W and the other native tasks intentionally share one runner so checkpoint
transfer, CSV diagnostics, and optimizer handling cannot diverge by task.
"""

from minidog_native.runner import CsvMetricsLogger, CsvMetricsRunner

__all__ = ("CsvMetricsLogger", "CsvMetricsRunner")

