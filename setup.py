from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np


# setup(
#     name="rotation_apply",
#     ext_modules=cythonize(
#         "rotations.pyx",
#         compiler_directives={"language_level": "3", "boundscheck": False, "wraparound": False}
#     ),
#     include_dirs=[np.get_include()],
# )

#_____________________________________
# extensions = [
#     Extension(
#         name="hkl_pg_generation",
#         sources=["hkl_pg_generation.pyx"],
#         extra_compile_args=["-O3", "-fopenmp"],
#         extra_link_args=["-fopenmp"],
#         include_dirs=[np.get_include()]
#     )
# ]
#
# setup(
#     name="hkl_pg_generation",
#     ext_modules=cythonize(
#         extensions,
#         compiler_directives={
#             "language_level": 3,
#             "boundscheck": False,
#             "wraparound": False,
#             "nonecheck": False,
#             "cdivision": True
#         }
#         )
#     ),

