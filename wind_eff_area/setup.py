from distutils.core import setup, Extension
import numpy as np
from distutils.core import setup
from Cython.Build import cythonize

setup(name="mid_point_loop", ext_modules=cythonize('mid_point_loop.pyx'),)
setup(name="geo_dist", ext_modules=cythonize('geo_dist.pyx'),)