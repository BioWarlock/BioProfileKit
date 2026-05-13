import sysconfig
import numpy as np
from Cython.Build import cythonize
from setuptools import setup, Extension, find_packages


def get_python_include_dir():
    return sysconfig.get_path('include')

extensions = [
    Extension(
        "cython_wrapper.wrapper_utils",
        ["src/cython_wrapper/wrapper_utils.pyx"],
    ),
    Extension(
        "cython_wrapper.taxonomy_validator",
        ["src/cython_wrapper/taxonomy_validator.pyx"],
    ),
    Extension("cython_wrapper.medcouple_fast",
              ["src/cython_wrapper/medcouple_fast.pyx"],
              include_dirs=[np.get_include()])
]

setup(
    name="BioProfileKit",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
    zip_safe=False,
)