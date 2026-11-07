"""
Granmo's Multi-Class Tsetlin Machine as uploaded on 21 April 2018 (the version
his Binary Iris demo was written for), compiled on first import with pyximport.
See MultiClassTsetlinMachine.pyx for the source and the two changes made to it.
"""

import numpy as np
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=2)

from . import MultiClassTsetlinMachine  # noqa: E402

__all__ = ["MultiClassTsetlinMachine"]
