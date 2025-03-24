from setuptools import setup
import os

from Cython.Build import cythonize
import numpy as np

PACKAGE_DIR = "PCLRC"

setup(
    name="PCLRC",
    version="0.1.0a",
    description="PCLRC: Probabilistic context likelihood of relatedness to "
                "build correlation networks and study the network "
                "differences.",
    author="Dong Nai-ping",
    author_email="nai-ping.dong@polyu.edu.hk",
    packages=[
        "PCLRC",
        "PCLRC.core",
    ],
    extra_compile_args=['-O3', '-mavx', '-ffast-math'],
    ext_modules=cythonize(
        [
            os.path.join(PACKAGE_DIR, "core/*.pyx")
        ],
        compiler_directives={
            "language_level": "3",
        }
    ),
    include_dirs=[
        np.get_include()
    ],
    include_package_data=True,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English"
    ],
)
