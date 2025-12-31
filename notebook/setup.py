"""
Setup script for NOMA Jupyter Magic extension.

Install with:
    pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="noma-jupyter-magic",
    version="0.1.0",
    description="IPython magic for executing NOMA code in Jupyter notebooks",
    author="NOMA Project",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "ipython>=7.0",
    ],
    entry_points={
        "IPython.extensions": [
            "noma_magic = noma_magic:load_ipython_extension",
        ],
    },
    classifiers=[
        "Framework :: IPython",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
    ],
)
