#!/usr/bin/env python
import io
import os

from os.path import join as pjoin
from setuptools import setup, find_packages


def get_version(file, name='__version__'):
    path = os.path.realpath(file)
    version_ns = {}
    with io.open(path, encoding="utf8") as f:
        exec(f.read(), {}, version_ns)
    return version_ns[name]


with open('README.md') as f:
    README = f.read()


with open('LICENSE') as f:
    LICENSE = f.read()

NAME = 'axisproj'
VERSION = get_version(pjoin(NAME, '_version.py'))

setup(
    name=NAME,
    version=VERSION,
    description='Optimal Axis-Aligned projections',
    long_description=README,
    author='Yarden Livnat',
    author_email='yarden@sci.utah.edu',
    url='https://github.com/yarden_livnat/axisproj',
    license=LICENSE,
    zip_safe=False,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        'numpy', 'sklearn', 'scipy', 'cvxpy', 'nudged'
    ],
    # test_suite='nose.collector',
    # tests_require=['nose'],
    entry_points={
        # 'console_scripts': [
        #     'regulus=regulus.command_line:main'
        # ],
    }
)
