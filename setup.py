import sys
import platform
import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize
import subprocess


# setup(
#     name="rotation_apply",
#     ext_modules=cythonize(
#         "rotation.pyx",
#         compiler_directives={
#             "language_level": "3",
#         }
#     ),
#     include_dirs=[np.get_include()],
# )

# _____________________________________
# extra_compile_args = []
# extra_link_args = []
# if sys.platform == "win32":
#     extra_compile_args = ["/openmp", "/O2", "/fp:fast"]
# else:
#     extra_compile_args = ["-fopenmp", "-O3"]
#     extra_link_args = ["-fopenmp"]
#
# extensions = [
#     Extension(
#         name="hkl_pg_generation",
#         sources=["hkl_pg_generation.pyx"],
#         extra_compile_args=extra_compile_args,
#         extra_link_args=extra_link_args,
#         include_dirs=[np.get_include()],
#         define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
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
#     ),
# )
# _____________________________________
#
# extra_compile_args = []
# libraries = []
#
# if sys.platform == "win32":
#     extra_compile_args = ["/openmp", "/O2", "/fp:fast"]
#     libraries = ["vcomp"]
# else:
#     extra_compile_args = ["-fopenmp", "-O3", "-march=native", "-ffast-math"]
#
# extensions = [
#     Extension(
#         name="vec_mx_rot",
#         sources=["vec_mx_rot.pyx"],
#         extra_compile_args=extra_compile_args,
#         libraries=libraries,
#         include_dirs=[np.get_include()],
#         define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
#     )
# ]
#
# setup(
#     ext_modules=cythonize(extensions, language_level=3),
# )



define_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]


def get_cpu_capabilities():
    """Определяет поддерживаемые инструкции процессора"""
    caps = set()

    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        caps = set(info.get('flags', []))
        # Для процессоров ARM
        if 'Features' in info:
            caps.update(info['Features'])
    except ImportError:
        pass

    # Дополнительные проверки
    try:
        # Проверка AVX-512 (может не определяться через cpuinfo)
        output = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.STDOUT)
        if "avx512" in output.lower():
            caps.add("avx512")
    except:
        pass

    return caps


# Получаем информацию о процессоре
cpu_caps = get_cpu_capabilities()
print(f"Detected CPU capabilities: {', '.join(cpu_caps)}")

extra_compile_args = []
libraries = []
extra_link_args = []
arch_flags = []

if sys.platform == "win32":
    extra_compile_args.extend([
        "/openmp",
        "/O2",
        "/fp:fast",
        "/Oi",
    ])
    libraries = ["vcomp"]

    if 'avx512' in cpu_caps:
        arch_flags = ["/arch:AVX512"]
    elif 'avx2' in cpu_caps:
        arch_flags = ["/arch:AVX2"]
    elif 'avx' in cpu_caps:
        arch_flags = ["/arch:AVX"]
    else:
        arch_flags = ["/arch:SSE2"]

else:
    extra_compile_args.extend([
        "-fopenmp",
        "-O3",
        "-ffast-math",
        "-funroll-loops",
        "-fno-strict-aliasing",
    ])
    extra_link_args = ["-fopenmp"]

    if 'avx512' in cpu_caps:
        arch_flags = ["-mavx512f", "-mavx512cd", "-mavx512bw", "-mavx512dq", "-mavx512vl"]
    elif 'avx2' in cpu_caps:
        arch_flags = ["-mavx2"]
    elif 'avx' in cpu_caps:
        arch_flags = ["-mavx"]
    elif 'sse4_2' in cpu_caps:
        arch_flags = ["-msse4.2"]
    elif 'sse2' in cpu_caps:
        arch_flags = ["-msse2"]
    elif platform.machine().startswith('arm'):
        if 'neon' in cpu_caps:
            arch_flags = ["-mfpu=neon", "-mfloat-abi=hard"]
        elif 'asimd' in cpu_caps:
            arch_flags = ["-march=armv8-a+simd"]

extra_compile_args.extend(arch_flags)

print(f"Using compiler flags: {extra_compile_args}")

extensions = [
    Extension(
        "circ_link_obst_i",
        sources=["circ_link_obst_i.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        libraries=libraries,
        include_dirs=[np.get_include()],
        define_macros=define_macros,
        language="c++",
    ),
    Extension(
        "circ_link_obst_n",
        sources=["circ_link_obst_n.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        libraries=libraries,
        include_dirs=[np.get_include()],
        define_macros=define_macros,
        language="c++",
    ),
    Extension(
        "rctngl_link_obst",
        sources=["rctngl_link_obst.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        libraries=libraries,
        include_dirs=[np.get_include()],
        define_macros=define_macros,
        language="c++",
    )
]

setup(
    name="obstacle_checks",
    ext_modules=cythonize(
        extensions,
        language_level="3str",
        compiler_directives={
            'binding': True,
            'embedsignature': True,
            'cdivision': True,
            'boundscheck': False,
            'wraparound': False,
            'nonecheck': False,
            'overflowcheck': False,
        }
    ),
    zip_safe=False,
    install_requires=[
        'cython',
        'numpy',
        'py-cpuinfo; platform_system != "Windows"'
    ]
)