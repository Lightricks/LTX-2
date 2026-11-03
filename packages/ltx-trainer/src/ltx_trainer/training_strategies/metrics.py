"""Cross-rank reduction of strategy-reported step metrics.
The trainer (kernel) owns turning a strategy's ``dict[str, float | MetricValue]`` into a single
``dict[str, float]`` ready for logging.
A plain ``float`` uses the common-case default: an **unweighted mean** across the ranks that
reported it. Wrap a metric in ``MetricValue`` to pick a different policy:
- ``mean``            — weighted mean, ``sum(value * weight) / sum(weight)``
- ``sum``             — sum of per-rank values
- ``min`` / ``max``   — extremum of per-rank values
- ``main_rank``       — the main rank's value only (no cross-rank combination)
- ``already_reduced`` — value the strategy reduced itself; taken as-is
Reduction runs behind a single ``gather_object`` collective so heterogeneous key sets across ranks
(e.g. sparse sigma buckets) never deadlock on per-key collectives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from accelerate.utils import gather_object

from ltx_trainer.training_strategies.types import MetricValue

if TYPE_CHECKING:
    from accelerate import Accelerator


def reduce_step_metrics(
    metrics: dict[str, float | MetricValue],
    accelerator: "Accelerator",
) -> dict[str, float]:
    """Reduce one rank's metrics across all ranks into plain floats ready for logging.
    Single-process runs skip the collective. Multi-process runs gather every rank's metric dict
    with one ``gather_object`` call, then reduce each metric over the ranks that reported it: plain
    floats become an unweighted mean; ``MetricValue`` metrics use their declared policy.
    """
    if accelerator.num_processes == 1:
        per_rank: list[dict[str, float | MetricValue]] = [metrics]
    else:
        per_rank = list(gather_object([metrics]))

    main = per_rank[0]

    # Union of keys across ranks, main-first, so sparse per-rank metrics (e.g. sigma buckets) and
    # metrics only some ranks report are all included exactly once.
    keys: list[str] = []
    seen: set[str] = set()
    for rank in per_rank:
        for key in rank:
            if key not in seen:
                seen.add(key)
                keys.append(key)

    reduced: dict[str, float] = {}
    for key in keys:
        entries = [rank[key] for rank in per_rank if key in rank]
        result = _reduce_metric(entries, main_value=main.get(key))
        if result is not None:
            reduced[key] = result
    return reduced


def _reduce_metric(entries: list[float | MetricValue], *, main_value: float | MetricValue | None) -> float | None:
    """Combine one metric's per-rank entries into a single float.
    Plain-float entries default to an unweighted mean. ``MetricValue`` entries use their (shared)
    reduction policy. Returns ``None`` only for a ``main_rank`` metric the main rank did not report.
    """
    metric_entries = [entry for entry in entries if isinstance(entry, MetricValue)]
    if not metric_entries:
        # Plain floats: unweighted mean across the ranks that reported the metric.
        return sum(float(entry) for entry in entries) / len(entries)

    reduction = metric_entries[0].reduction
    if reduction in ("main_rank", "already_reduced"):
        if isinstance(main_value, MetricValue):
            return float(main_value.value)
        # `already_reduced` is identical across ranks, so any rank's value is authoritative;
        # a `main_rank` metric the main rank did not report has nothing to log.
        return float(metric_entries[0].value) if reduction == "already_reduced" else None

    values = [entry.value for entry in metric_entries]
    aggregators = {"sum": sum, "min": min, "max": max}
    if reduction in aggregators:
        return float(aggregators[reduction](values))

    # Weighted mean (default). Fall back to an unweighted mean when all weights are zero.
    total_weight = sum(entry.weight for entry in metric_entries)
    weighted = sum(entry.value * entry.weight for entry in metric_entries)
    return float(weighted / total_weight) if total_weight else float(sum(values) / len(values))
