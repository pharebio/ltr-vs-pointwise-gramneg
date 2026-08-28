"""
Organism-level task groupings for gram-negative evaluation and the
group-conditioned ranker.

The mapping from public assay endpoints to organism groups is shipped as
``data/organism_groups.json``. The groups separate biologically incompatible
phenotypes on purpose -- e.g. ``e_coli_wt`` vs ``e_coli_efflux_ko`` (tolC) vs
``e_coli_om_defect`` (lpxC) -- which is why we never form ranking pairs across
groups and why a single pan-GN score muddies them.
"""
from __future__ import annotations

import json
import os

# Shipped, repo-local grouping file (public endpoints only).
DEFAULT_ORGANISM_GROUPS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "organism_groups.json")


def load_organism_groups(path: str = DEFAULT_ORGANISM_GROUPS_PATH) -> dict[str, list[str]]:
    """Return ``{group_key: [task_name, ...]}`` from the grouping JSON."""
    with open(path) as fh:
        data = json.load(fh)
    return {g: list(v["tasks"]) for g, v in data["groups"].items()}


def task_to_group(path: str = DEFAULT_ORGANISM_GROUPS_PATH) -> dict[str, str]:
    """Inverse map ``{task_name: group_key}`` for tagging per-task metric rows."""
    out: dict[str, str] = {}
    for group, tasks in load_organism_groups(path).items():
        for t in tasks:
            out[t] = group
    return out
