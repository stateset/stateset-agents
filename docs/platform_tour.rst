Platform Tour
=============

The canonical guided walk through the platform — from ``pip install`` to a
published v1.0 whitepaper revision — lives in ``PLATFORM_TOUR.md`` at the
repo root.

We keep it in Markdown so it renders well on GitHub and pastes cleanly into
PRs and Slack threads. To read it:

* **On GitHub:** `docs/PLATFORM_TOUR.md <https://github.com/stateset/stateset-agents/blob/master/docs/PLATFORM_TOUR.md>`_
* **From the CLI:** ``stateset-agents tour`` (uses ``$PAGER`` if available)
* **From a clone:** ``cat docs/PLATFORM_TOUR.md`` or open in any editor

What the tour covers
--------------------

1. ``pip install`` — lean core + opt-in extras
2. Scaffold a project with ``stateset-agents starter``
3. Edit and train with the bundled GSPO trainer
4. Benchmark via the Phase 0 pipeline
5. Aggregate, plot, publish
6. Serve with ``stateset-agents serve --checkpoint``
7. Local CI parity with ``make smoke``
8. The three Colab notebooks for the three pillars
9. The full whitepaper
10. Close the loop: curate → SFT → iterate
11. Where to go next

The FAQ at the end of the tour answers the seven questions that actually come
up running this — checkpoint not loading, no GPU detected, GSPO clip-range
surprises, custom datasets, custom models, etc.

Live demo
---------

For a recorded demonstration of the full pipeline, run ``make demo`` from a
checkout. It runs end-to-end in ~3 seconds without GPU and produces real
artifacts. Designed to be screen-recorded or screen-shared.
