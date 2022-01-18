from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_desc = f.read()

with open(path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    install_requires = f.read()

setup(
    name = "bayesian-torch",
    packages = find_packages(),
    version = "0.2.0",
    description = "Bayesian layers and utilities to perform stochastic variational inference in PyTorch",
    author = "ranganath.krishnan@intel.com",
    url = "https://github.com/IntelLabs/bayesian-torch",
    long_description = long_desc,
    long_description_content_type = "text/markdown",
    install_requires = install_requires,
    classifiers = [
                    "Development Status :: 3 - Alpha",
                    "Intended Audience :: Developers",
                    "Intended Audience :: Science/Research",
                    "Programming Language :: Python :: 3.7"
                  ]
)
