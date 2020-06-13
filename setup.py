import os
import sys
import glob
import subprocess

from setuptools import setup, Extension
from Cython.Build import cythonize

# define helper functions to manage the build/install
# ---------------------------------------------------

def find_gcc():
    """locate the actual GCC binary if it has been aliased"""

    patterns = [
        "/usr/bin/gcc-[0-9]",
        "/opt/bin/gcc-[0-9]",
        "/usr/local/bin/gcc-[0-9]",
        "/opt/local/bin/gcc-[0-9]"
    ]

    binaries = []
    for pattern in patterns:
        binaries += glob.glob(pattern)

    if binaries:
        binary = binaries[-1]
        path, gcc = os.path.split(binary)
        return gcc
    else:
        raise ImportError('cannot locate GCC binary - install via homebrew or linux package manager')


def check_gcc():
    """ensure GCC is installed on the user's system"""

    if sys.platform == 'darwin':
        gcc = find_gcc()
    elif sys.platform == 'linux':
        gcc = 'gcc'
    else:
        raise OSError('RankFM currently only supports OSX and Linux')

    try:
        subprocess.run("{gcc} --version".format(gcc=gcc), shell=True, check=True)
        os.environ["CC"] = gcc
    except subprocess.CalledProcessError:
        raise ImportError('cannot locate/execute GCC binary - install via homebrew or linux package manager')

# define the extension packages to include
# ----------------------------------------

### "-O3", "-ffast-math", "-march=native",
disabled_warnings = ['-Wno-unused-function', '-Wno-maybe-uninitialized', '-Wno-alloc-size-larger-than']
compile_args = ['-fopenmp'] + disabled_warnings
link_args = ['-fopenmp']

extensions = [
    Extension(
        name='rankfm._rankfm',
        sources=['rankfm/_rankfm.pyx', 'rankfm/mt19937ar/mt19937ar.c'],
        extra_compile_args=compile_args,
        extra_link_args=link_args
    )
]

# define the main package setup function
# --------------------------------------

setup(
    name='rankfm',
    version='0.2.3',
    description='a python implementation of the generic factorization machines model class '
                'adapted for collaborative filtering recommendation problems '
                'with implicit feedback user-item interaction data '
                'and (optionally) additional user/item side features',
    author='Eric Lundquist',
    author_email='e.t.lundquist@gmail.com',
    url='https://github.com/etlundquist/rankfm',
    keywords=['machine', 'learning', 'recommendation', 'factorization', 'machines', 'implicit'],
    license='GNU General Public License v3.0',
    packages=['rankfm'],
    ext_modules=cythonize(extensions, annotate=True),
    zip_safe=False,
    python_requires='>=3.6',
    install_requires=['numpy>=1.15', 'pandas>=0.24', 'Cython>=0.29']
)

