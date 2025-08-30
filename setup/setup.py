import sys
import platform
import argparse
import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize


def get_compilation_args(module_name):
    """Get individual compilation arguments for each module"""
    define_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]

    # Base settings for Windows
    if sys.platform == "win32":
        base_compile_args = ["/O2", "/openmp", "/Oi"]
        libraries = ["vcomp"]
        extra_link_args = []
    else:
        base_compile_args = [
            "-O2", "-fopenmp", "-fno-math-errno",
            "-fno-strict-aliasing", "-march=x86-64"
        ]
        libraries = []
        extra_link_args = ["-fopenmp"]

    # Individual settings for each module
    module_settings = {
        'rotation_apply': {
            'extra_compile_args': base_compile_args + (["/fp:precise"] if sys.platform == "win32" else []),
            'extra_link_args': extra_link_args,
            'libraries': libraries
        },
        'hkl_pg_generation': {
            'extra_compile_args': base_compile_args + (["/fp:precise"] if sys.platform == "win32" else []),
            'extra_link_args': extra_link_args,
            'libraries': libraries
        },
        'vec_mx_rot': {
            'extra_compile_args': base_compile_args + (["/fp:precise"] if sys.platform == "win32" else []),
            'extra_link_args': extra_link_args,
            'libraries': libraries
        },
        'circ_link_obst_i': {
            'extra_compile_args': base_compile_args + (["/fp:fast"] if sys.platform == "win32" else []),
            'extra_link_args': extra_link_args,
            'libraries': libraries,
            'language': "c++"
        },
        'circ_link_obst_n': {
            'extra_compile_args': base_compile_args + (["/fp:fast"] if sys.platform == "win32" else []),
            'extra_link_args': extra_link_args,
            'libraries': libraries,
            'language': "c++"
        },
        'rctngl_link_obst': {
            'extra_compile_args': base_compile_args + (["/fp:fast"] if sys.platform == "win32" else []),
            'extra_link_args': extra_link_args,
            'libraries': libraries,
            'language': "c++"
        },
        'calc_lorentz_coefficient': {
            'extra_compile_args': base_compile_args + (["/fp:precise"] if sys.platform == "win32" else []),
            'extra_link_args': extra_link_args,
            'libraries': libraries,
            'language': "c++"
        }
    }

    return {**module_settings.get(module_name, {}), 'define_macros': define_macros}


def get_all_extensions(selected_modules=None):
    """Get extensions for selected modules"""
    all_modules = {
        'rotation_apply': {
            'sources': ["rotation.pyx"],
            'include_dirs': [np.get_include()]
        },
        'hkl_pg_generation': {
            'sources': ["hkl_pg_generation.pyx"],
            'include_dirs': [np.get_include()]
        },
        'vec_mx_rot': {
            'sources': ["vec_mx_rot.pyx"],
            'include_dirs': [np.get_include()]
        },
        'circ_link_obst_i': {
            'sources': ["circ_link_obst_i.pyx"],
            'include_dirs': [np.get_include()]
        },
        'circ_link_obst_n': {
            'sources': ["circ_link_obst_n.pyx"],
            'include_dirs': [np.get_include()]
        },
        'rctngl_link_obst': {
            'sources': ["rctngl_link_obst.pyx"],
            'include_dirs': [np.get_include()]
        },
        'calc_lorentz_coefficient': {
            'sources': ["calc_lorentz_coefficient.pyx"],
            'include_dirs': [np.get_include()]
        }
    }

    extensions = []
    modules_to_build = selected_modules if selected_modules else all_modules.keys()

    for module_name in modules_to_build:
        if module_name in all_modules:
            compilation_args = get_compilation_args(module_name)
            extensions.append(Extension(
                name=module_name,
                **all_modules[module_name],
                **compilation_args
            ))
            print(f"Added module: {module_name}")
        else:
            print(f"Warning: module {module_name} not found")

    return extensions


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Cython modules compilation',
                                     add_help=False)
    parser.add_argument('-a', '--all', action='store_true',
                        help='Compile all modules')
    parser.add_argument('-m', '--modules', nargs='+',
                        choices=['rotation_apply', 'hkl_pg_generation', 'vec_mx_rot',
                                 'circ_link_obst_i', 'circ_link_obst_n', 'rctngl_link_obst',
                                 'calc_lorentz_coefficient'],
                        help='Compile selected modules')
    parser.add_argument('--list', action='store_true',
                        help='Show available modules list')
    parser.add_argument('--help-modules', action='store_true',
                        help='Show modules help')

    return parser.parse_known_args()


def main():
    # Parse arguments, pass remaining to setup()
    print("Original sys.argv:", sys.argv)
    args, remaining_argv = parse_arguments()
    print('Parsed args:', args)
    print('Remaining argv:', remaining_argv)

    if args.list:
        print("Available modules for compilation:")
        for module_name in ['rotation_apply', 'hkl_pg_generation', 'vec_mx_rot',
                            'circ_link_obst_i', 'circ_link_obst_n', 'rctngl_link_obst',
                            'calc_lorentz_coefficient']:
            print(f"  - {module_name}")
        return

    if args.help_modules:
        print("Modules help:")
        print("  rotation_apply           - /fp:precise")
        print("  hkl_pg_generation        - /fp:precise")
        print("  vec_mx_rot               - /fp:precise")
        print("  circ_link_obst_i         - /fp:fast, C++")
        print("  circ_link_obst_n         - /fp:fast, C++")
        print("  rctngl_link_obst         - /fp:fast, C++")
        print("  calc_lorentz_coefficient - /fp:precise, C++")
        return

    if args.all:
        # Compile all modules
        selected_modules = None
        print("Compiling all modules...")
    elif args.modules:
        # Compile selected modules
        selected_modules = args.modules
        print(f"Compiling selected modules: {', '.join(selected_modules)}")
    else:
        print("Error: must specify either -a/--all or -m/--modules")
        print("Use --help for usage information")
        return

    selected_extensions = get_all_extensions(selected_modules)

    if not selected_extensions:
        print("No modules to compile")
        return

    # Set remaining arguments for setup()
    # Ensure we have the necessary commands for setuptools
    if not any(arg in remaining_argv for arg in ['build', 'build_ext', 'install']):
        remaining_argv = ['build_ext', '--inplace'] + remaining_argv

    print("Final sys.argv:", [sys.argv[0]] + remaining_argv)
    sys.argv = [sys.argv[0]] + remaining_argv

    setup(
        name="cython_modules",
        ext_modules=cythonize(
            selected_extensions,
            compiler_directives={
                "language_level": "3",
                "boundscheck": False,
                "wraparound": False,
                "nonecheck": False,
                "cdivision": True
            }
        ),
        include_dirs=[np.get_include()],
        zip_safe=False,
    )


if __name__ == "__main__":
    main()