"""Where the frozen generations live.

Resolved through one function rather than hard-coded, for two reasons that both matter:

* **The tests must not write into the tree they are scoring.** A test that drops a calibrated
  generation into `eval/cells/` pollutes the repository, and an interrupted run leaves it there --
  where the next `burnish generation show` will happily find it and a contributor will wonder why
  a cell they never calibrated has numbers in it.
* **The trusted runner stages the instrument to a temporary directory.** `eval/run_from_base.sh`
  overlays `eval/` from the base commit, so the generations it scores against have to resolve
  relative to the staged copy and not to the working tree.

`BURNISH_CELLS_ROOT` overrides the default for a process and everything it spawns.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT = Path(__file__).resolve().parent / "cells"


def cells_root(explicit=None) -> Path:
    """Explicit argument, then the environment, then the tree."""
    if explicit:
        return Path(explicit)
    env = os.environ.get("BURNISH_CELLS_ROOT")
    return Path(env) if env else DEFAULT


def generation_path(name, explicit=None) -> Path:
    return cells_root(explicit) / name / "generation.json"


def add_argument(parser):
    parser.add_argument("--cells-root",
                        help="directory holding the frozen generations "
                             "(default: eval/cells, or $BURNISH_CELLS_ROOT)")
