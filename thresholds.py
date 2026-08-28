"""
Per-endpoint hit-threshold resolution for LODO evaluation.

Real gram-negative assays do NOT share a single hit cutoff: %growth-inhibition
tasks call a hit at >= 80% GI, while the continuous pMIC task calls a hit at
pMIC >= 7 (MIC <= 100 nM). This can be recorded per endpoint in a metadata JSON
(keyed by task / endpoint name, each entry carrying a ``"threshold"`` field).

Passing that map -- or any ``{dataset_id: threshold}`` dict, or a callable -- as
``active_threshold`` to ``evaluate.run_lodo`` / ``compare_to_pointwise_baseline``
binarizes each held-out dataset with its OWN cutoff, instead of one global scalar
that can leave an assay with zero actives (-> NaN BEDROC/EF and a silently skipped
fold). A scalar is still accepted for the single-assay / synthetic case.

Pure Python (no numpy/torch) so it stays importable everywhere, including the
torch-free test suite and CI.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping

# What ``active_threshold`` may be throughout the harness.
ThresholdSpec = float | int | Mapping | Callable[[str], float | None] | None


def load_endpoint_thresholds(source) -> dict[str, float]:
    """
    Load a ``{endpoint_name: threshold}`` map.

    ``source`` is either a path to a JSON file or an already-loaded dict. Handles
    both the nested metadata shape (``{task: {"threshold": x, "dataset": ...}}``)
    and a flat ``{task: threshold}`` mapping. Entries without a resolvable
    threshold are skipped.
    """
    if isinstance(source, (str, bytes)) or hasattr(source, "__fspath__"):
        with open(source) as fh:
            data = json.load(fh)
    else:
        data = dict(source)

    out: dict[str, float] = {}
    for task, val in data.items():
        if isinstance(val, Mapping):
            if "threshold" in val and val["threshold"] is not None:
                out[str(task)] = float(val["threshold"])
        elif val is not None:
            out[str(task)] = float(val)
    return out


def resolve_threshold(active_threshold: ThresholdSpec, key) -> float | None:
    """
    Resolve the hit threshold for one dataset / group id ``key``.

    ``active_threshold`` may be:
        * ``None``       -> ``None`` (early-recognition metrics are skipped)
        * ``float``/``int`` -> used for every dataset (legacy scalar behavior)
        * ``Mapping``    -> looked up by ``key`` (``None`` if the key is absent)
        * ``Callable``   -> called with ``key``
    """
    if active_threshold is None or isinstance(active_threshold, (int, float)):
        return active_threshold
    if isinstance(active_threshold, Mapping):
        thr = active_threshold.get(str(key), active_threshold.get(key))
        return None if thr is None else float(thr)
    if callable(active_threshold):
        return active_threshold(key)
    raise TypeError(f"unsupported active_threshold type: {type(active_threshold)!r}")
