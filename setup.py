from setuptools import setup

setup(
    name="fuzzyops",
    version="1.0.0",
    description="fuzzy math library",
    long_description="file: README.md",
    long_description_content_type="text/markdown",
    classifiers=["Programming Language :: Python :: 3",
                 "Operating System :: OS Independent"],
    python_requires=">=3.10",
    install_requires=["setuptools>=60.2.0", "matplotlib>=3.6.2"],
    extras_require={
        "win": ["torch --index-url https://download.pytorch.org/whl/cu117"],
        "macos": ["torch"],
        "unix": ["torch"]
    }
)