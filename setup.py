import glob
from setuptools import Extension, setup

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

# add compiler arguments to optimize machine code and ignore warnings
disabled_warnings = ['-Wno-unused-function', '-Wno-uninitialized']
compile_args = ['-O2', '-ffast-math'] + disabled_warnings

# define the _rankfm extension including the wrapped MT module
extensions = [
    Extension(
        name='rankfm._rankfm',
        sources=['rankfm/_rankfm.{ext}'.format(ext=ext), 'rankfm/mt19937ar/mt19937ar.c'],
        extra_compile_args=compile_args,
        # extra_compile_args={'gcc': ['/Qstd=c99']},
    )
]

# re-generate the C code if needed
if use_cython:
    extensions = cythonize(extensions)

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# define the main package setup function
# --------------------------------------
setup(
    name='RankFM',
    version='0.2.6',
    description='a python implementation of the generic factorization machines model class '
                'adapted for collaborative filtering recommendation problems '
                'with implicit feedback user-item interaction data '
                'and (optionally) additional user/item side features',
    author='ErraticO',
    author_email='wyh123132@163.com',
    url='https://github.com/ErraticO/rankfm',
    keywords=['machine', 'learning', 'recommendation', 'factorization', 'machines', 'implicit'],
    license='GNU General Public License v3.0',
    packages=['RankFM'],
    ext_modules=extensions,
    zip_safe=False,
    python_requires='>=3.9',
    install_requires=['numpy>=1.15', 'pandas>=1.5'],
    long_description=long_description,
    long_description_content_type="text/markdown",
)

