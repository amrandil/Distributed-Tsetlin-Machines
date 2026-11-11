"""
Granmo's original 2018 Multi-Class Tsetlin Machine (Cython), compiled on first
import with pyximport. Needs Cython and a C compiler; the compiled module is
cached in ~/.pyxbld, outside the repository.

See MultiClassTsetlinMachine.pyx for the source and the two changes made to it.
"""

import numpy as np
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=2)

from . import MultiClassTsetlinMachine  # noqa: E402
from . import TsetlinMachine  # noqa: E402  (single-output TM, used for Game of Life)

__all__ = ["MultiClassTsetlinMachine", "TsetlinMachine"]
