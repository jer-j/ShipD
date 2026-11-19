import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "shipd_jax.reference.new_modmichell",
        sources=["shipd_jax/reference/new_modmichell.pyx"],
    ),
    Extension(
        "shipd_jax.reference.hull_parameterization",
        sources=["shipd_jax/reference/hull_parameterization.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        language_level=3,
        compiler_directives={
            "boundscheck": False,
            # the reference code indexes from the end (x[-1]); wraparound=False
            # turns those into out-of-bounds errors
            "cdivision": True,
            "initializedcheck": False,
        },
    ),
)
