import sys

from setuptools import setup

from pathlib import Path


def read_requirements(path):
    return list(Path(path).read_text().splitlines())


reqs = read_requirements("requirements.txt")

setup(
    name="fuzzyops",
    version="1.0.0",
    description="fuzzy math library",
    long_description="file: README.md",
    long_description_content_type="text/markdown",
    classifiers=["Programming Language :: Python :: 3",
                 "Operating System :: OS Independent"],
    python_requires=">=3.10",
    install_requires=reqs

)
