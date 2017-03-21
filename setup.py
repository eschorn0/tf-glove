from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'Build Co-occurrence matrix',
  ext_modules = cythonize("buildHashMatrix.pyx"),
)
