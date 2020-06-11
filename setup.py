from setuptools import setup, Extension
from Cython.Build import cythonize

# define the extension packages to include
# ----------------------------------------

extensions = [
    Extension("rankfm._rankfm", ["rankfm/_rankfm.pyx"])
]

# define the main package setup function
# --------------------------------------

setup(
    name='rankfm',
    version='0.2.0',
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
    ext_modules=cythonize(extensions),
    zip_safe=False,
    python_requires='>=3.6',
    install_requires=['numpy>=1.15', 'pandas>=0.24', 'Cython>=0.29']
)

