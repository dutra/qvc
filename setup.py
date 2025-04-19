#!/usr/bin/env python
import os
import sys
from setuptools import setup

from src.pyqsofit.version import __version__


# Load the __version__ variable without importing the package already

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='EzTaoX',
      version=__version__,
      description="",
      long_description=long_description,
      author='',
      author_email='',
      url='',
      license='GNU General Public License v3.0',
      package_dir={'eztaox': 'src/eztaox'},
      packages=['eztaox'],
      install_requires=install_requires,
      include_package_data=True,
      classifiers=[
          "License :: OSI Approved :: GNU License",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
          "Intended Audience :: Science/Research",
          "Topic :: Scientific/Engineering :: Astronomy",
          ],
      )
