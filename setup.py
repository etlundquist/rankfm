import glob
import sys
from pathlib import Path

import numpy
from setuptools import Extension, setup

NAME = 'rankfm'
VERSION = '0.2.6'

# define the extension packages to include
# ----------------------------------------

# prefer the generated C extensions when building
if glob.glob('rankfm/_rankfm.c'):
    print("building extensions with pre-generated C source...")
    use_cython = False
    ext = 'c'
else:
    print("re-generating C source with cythonize...")
    from Cython.Build import cythonize
    use_cython = True
    ext = 'pyx'

# add compiler and linker arguments to optimize machine code and ignore warnings
if sys.platform == "linux":
    compile_args = ['-O2', '-ffast-math', '-fopenmp', '-Wno-unused-function', '-Wno-uninitialized']
    link_args = ['-fopenmp']
elif sys.platform == "darwin":
    compile_args = [
        '-std=c99',
        '-O3',
    ]
    link_args = [
        '-I/opt/homebrew/opt/libomp/include',
        '-L/opt/homebrew/opt/libomp/lib',
        '-lomp'
    ]
else:
    compile_args = ['/openmp', '/O2']
    link_args = ['/openmp']
# define the _rankfm extension including the wrapped MT module
extensions = [
    Extension(
        name='rankfm._rankfm',
        sources=['rankfm/_rankfm.{ext}'.format(ext=ext), 'rankfm/mt19937ar/mt19937ar.c'],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        include_dirs=[numpy.get_include()],
    )
]

# re-generate the C code if needed
if use_cython:
    extensions = cythonize(extensions)

# read the contents of your README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# define the main package setup function
# --------------------------------------
setup(
    name=NAME,
    version=VERSION,
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
    ext_modules=extensions,
    zip_safe=False,
    python_requires='>=3.8',
    install_requires=['numpy>=1.15', 'pandas>=0.24'],
    long_description=long_description,
    long_description_content_type="text/markdown",
)
