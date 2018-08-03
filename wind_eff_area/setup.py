from distutils.core import setup
from Cython.Build import cythonize

setup(name="mid_point_loop", ext_modules=cythonize('mid_point_loop.pyx'),)