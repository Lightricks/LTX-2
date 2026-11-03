"""Shared, strategy-agnostic validation toolkit.
Submodules:
- ``config``: validation *content* config (samples/conditions/sampling settings), embedded by a
  strategy's config. A leaf module with no dependency on ``ltx_trainer.config`` or a strategy.
- ``runner``: the ``ValidationRunner`` sampler, which also owns distributed orchestration
  (``run`` + the ``compute_work_items`` staticmethod).
This package intentionally imports nothing heavy at the package level: ``runner`` pulls in
ltx-core inference modules, so callers import it explicitly (often lazily) rather than via the
package root.
"""
