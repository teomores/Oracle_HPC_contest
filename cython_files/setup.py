from setuptools import setup
from Cython.Build import cythonize
import numpy
from tqdm import tqdm

setup(
    ext_modules = cythonize("fast_extraction.pyx", annotate=True),
    include_dirs=[numpy.get_include()]
)