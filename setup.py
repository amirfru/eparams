"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.
setup(
    name='eparams',
    description='Parameter class for all',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/amirfru/eparams',
    author='Amir Fruchtman',
    author_email='amir.fru@gmail.com',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development',

        # Pick your license as you wish
        'License :: OSI Approved :: The Unlicense (Unlicense)',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "Programming Language :: Python :: 3.10",
    ],
    keywords='params, config, development',
    package_dir={'': '.'},
    packages=['eparams'],
    python_requires='>=3.6, <4',
    install_requires=['typeguard'],
)
