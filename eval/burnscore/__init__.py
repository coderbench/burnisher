"""burnscore -- the measuring instrument.

Nothing in this package runs a model. It turns raw paired measurements into a score, and it is
deliberately separable from the runtime for one reason: `eval/run_from_base.sh` overlays this
directory from the BASE commit before scoring a submission, so a candidate cannot win by editing
the ruler. A one-line change to a noise floor, a confidence level, a normalization bound or a
frozen generation does not look like cheating in a diff.
"""
from . import (bootstrap, cells, dtypes, floor, frontier, geometry, ledger, pipeline,
               receipt, roofline)

__all__ = ["bootstrap", "cells", "dtypes", "floor", "frontier", "geometry", "ledger",
           "pipeline", "receipt", "roofline"]
__version__ = "0.1.0"
