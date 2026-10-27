"""Product verticals built on the framework: refund and industry studies.

These modules compose every framework layer -- environments, datasets,
evaluation, trainers -- and the remote River runtime into end-to-end studies,
so they sit above ``training`` and ``remote`` and no framework layer may
import them (docs/ARCHITECTURE.md, Layering):

* ``river_refund`` -- the River-native refund RL experiment
  (``python -m stateset_agents.verticals.river_refund``);
* ``agent_study`` / ``study_preflight`` -- four-arm River refund study planning;
* ``acceptance`` -- A+ native acceptance revalidation;
* ``industry_evaluation`` / ``industry_synthetic`` -- industry fine-tuning
  evaluation and synthetic datasets.

Each moved here in v0.70.0; the old ``evaluation``/``data``/``training`` paths
are deprecated shims removed in v0.71.0. Import the submodules directly; this
package re-exports nothing so importing it stays free.
"""

from __future__ import annotations

__all__: list[str] = []
