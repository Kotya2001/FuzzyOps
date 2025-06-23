# FuzzyOps
Library of algorithms for fuzzy forecasting and decision support

The library is intended for use:
- in scientific laboratories engaged in research in the field of multi-criteria analysis, optimal planning and management;
- in companies engaged in the development of decision support systems. In fact, the library should be used in the creation of both full-featured software products and experimental mock-ups of software systems designed to work with fuzzy factors.

The library can also be used by directly calling functions in C++ programs, following the instructions:
- https://github.com/Kotya2001/FuzzyOps/blob/main/cpp/README.md


### How to install the library

To install the library as a pip package, use
the command: `pip install git+https://{login}:{token}@github.com/Kotya2001/FuzzyOps.git `
by substituting the appropriate values:

 - login: your login on GitHub
 - token: how to create a token - [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)

Or
 - ```pip install fuzzyops```

### Before installation

Create a virtual environment with Python >= 3.10

  ```Full path to the Python 3.10 executable file -m venv env```

Activating the environment

  - Macos: ```source env/bin/activate```
  - Windows: ```.\env\Scripts\activate```
  - Linux: ```source env/bin/activate```

Installing the Cuda Toolkit 11.5

  - https://developer.nvidia.com/cuda-11-5-0-download-archive

Install PyTorch depending on your operating system

  - Windows: ```pip3 install torch --index-url https://download.pytorch.org/whl/cu117```
  - Macos: ```pip3 install torch```
  - Linux: ```pip3 install torch```

### Minimum technical requirements

- RAM capacity of at least 2 GB;
- For CUDA calculations, an Nvidia GeForce RTX 3060 or higher graphics output device
- Installed Python version 3.10 or higher

### Instructions for using the library and documentation for the library's source code:

-  Instructions for working with the library - https://github.com/Kotya2001/FuzzyOps/wiki/Instructions-for-using-the-FuzzyOps-library;
-  Documentation for the library source code - https://fuzzyops.readthedocs.io/en/latest/

### Running tests

After installation, the tests are run according to the instructions.:

 - Instructions for running tests - https://github.com/Kotya2001/FuzzyOps/wiki/Instructions-for-running-FuzzyOps-library-tests
   

### Instructions for using the library in C++ programs

-  Instructions for using the library in C++ programs - https://github.com/Kotya2001/FuzzyOps/blob/main/cpp/README.md


### Source files structure

 * [cpp](https://github.com/Kotya2001/FuzzyOps/tree/main/cpp) - Instructions for using the library in C++ programs and examples of using the library in Python and C++;
 * [example](https://github.com/Kotya2001/FuzzyOps/tree/main/examples):
   * [common](https://github.com/Kotya2001/FuzzyOps/tree/main/examples/common) - Examples of using the library code;
   * The remaining files are practical examples of using the library code;
 * [src](https://github.com/Kotya2001/FuzzyOps/tree/main/src) - Library source codes:
   * [docs](https://github.com/Kotya2001/FuzzyOps/tree/main/src/docs) - Files, format .html with documentation for the source code (compiled using the library [sphinx](https://www.sphinx-doc.org/en/master/));
   * [fuzzyops](https://github.com/Kotya2001/FuzzyOps/tree/main/src/fuzzyops) - Library source codes:
     * [fan](https://github.com/Kotya2001/FuzzyOps/tree/main/src/fuzzyops/fan) - Source codes of fuzzy analytical networks;
     * [fuzzy_logic](https://github.com/Kotya2001/FuzzyOps/tree/main/src/fuzzyops/fuzzy_logic) Source codes of fuzzy logic algorithms;
     * [fuzzy_msa](https://github.com/Kotya2001/FuzzyOps/tree/main/src/fuzzyops/fuzzy_msa) - Source codes of classical multicriteria analysis algorithms with fuzzy variables;
     * [fuzzy_nn](https://github.com/Kotya2001/FuzzyOps/tree/main/src/fuzzyops/fuzzy_nn) - Source codes of algorithms for fuzzy neural networks (ANFIS Network);
     * [fuzzy_numbers](https://github.com/Kotya2001/FuzzyOps/tree/main/src/fuzzyops/fuzzy_numbers/fuzzify) - Source codes for implementing fuzzy numbers (fuzzification, defuzzification, fuzzy arithmetic);
     * [fuzzygraphs](https://github.com/Kotya2001/FuzzyOps/tree/main/src/fuzzyops/graphs/fuzzgraph) - Source codes for the implementation of fuzzy graphs;
     * [fuzzygraphs_algs](https://github.com/Kotya2001/FuzzyOps/tree/main/src/fuzzyops/graphs/algorithms) - Source codes of algorithms on fuzzy graphs (Fuzzy dominance relations, fuzzy factor models, fuzzy transport graphs);
     * [fuzzy_pred](https://github.com/Kotya2001/FuzzyOps/tree/main/src/fuzzyops/prediction) - Source codes of fuzzy prediction algorithms;
     * [sequencing_assignment](https://github.com/Kotya2001/FuzzyOps/tree/main/src/fuzzyops/sequencing_assignment) - The source codes of algorithms on fuzzy graphs of the sequence of work in assignment tasks;
     * [tests](https://github.com/Kotya2001/FuzzyOps/tree/main/src/fuzzyops/tests) - Algorithm Test Codes.
 * [readthedocs](https://github.com/Kotya2001/FuzzyOps/blob/main/.readthedocs.yml) - A file for automatic assembly and placement of documentation on https://about.readthedocs.com;
 * [doc_reqs.txt](https://github.com/Kotya2001/FuzzyOps/blob/main/doc_reqs.txt) - A library dependency file for building documentation https://about.readthedocs.com;
 * [requirements](https://github.com/Kotya2001/FuzzyOps/blob/main/requirements.txt) - The dependency file for installing the library;
 * [setup.cfg](https://github.com/Kotya2001/FuzzyOps/blob/main/setup.cfg) - Configuration file for building the library distribution;
 * [setup.py](https://github.com/Kotya2001/FuzzyOps/blob/main/setup.py) - A file for building a library distribution using `setuptools`;
 * [LICENSE](https://github.com/Kotya2001/FuzzyOps/blob/main/LICENSE) - Library license file;
   
