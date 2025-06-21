# Instructions for using the library in C++ code
Using the Python C/C++ API (https://docs.python.org/3/c-api/index.html ) it is possible to make a direct call to all classes and functions of the fuzzyops library in C++ code

The principle of operation is as follows:

It is necessary to install Python version 3.10 and higher in the system (Linux, Windows, Macos, the principle is the same)
(you can also use anaconda to create an environment https://www.anaconda.com rather than using the system)

The folders with files with Python installed in the include directory contain the Python.h header file, and the lib directory contains dynamic libraries.

By including it in a C++ project (#include <Python.h>), compiling the C++ code with the following flags:
	
- The -I flag is used to specify additional directories in which the compiler will search for header files (.h).
For example, we have this path to the directory include
/usr/include/python3.10 indicates to the g++ compiler that it should search for header files such as Python.h in the specified directory. 
(You need to find the directory with the Python header files, usually the include directory)

- The -L flag indicates the paths in which the compiler will search for dynamic
libraries (.so or .a files) when linking the program.
For example, we have this path to the lib directory with the files /usr/lib/python3.10/config-3.10-x86_64-linux-gnu tells the compiler to look for libraries in this directory (the /usr/lib directory).
(You need to find the directory with Python libraries, usually the lib directory)

- The -l flag tells the compiler which specific library to link to the program. After the -l, the library name is specified without the lib prefix and without the extension
For example, -lpython3.10 tells the compiler to link to the libpython3.10.so (or similar) library to use functions defined in the Python API.

By pre-installing the fuzzyops library either in the system environment or in the anaconda environment (if you are using it), you can directly call functions and classes from the library in your C++ program.

## Directory description

The directory contains examples of Python code for the fuzzyops library and their corresponding C++ code examples (how to call library functions and classes), as well as an example Makefile with which you can assemble and test code execution. 
The C++ examples use typical functions from
Python.however, using other modules of the fuzzyops library in C++ code is done in a similar way, therefore, in your C++ projects using the fuzzyops library, you should use the documentation for the library code () and
the Python C/C++ API documentation (https://docs.python.org/3/c-api/index.html )


## Example of installing the library and running code samples in C++

Here is an example of installing the library and running a C++ example code for Ubuntu 22.04.5

### №1. Before installation 

Check if the g++ and gcc compilers are installed
(You can use version 9 or higher, as long as it includes the C++11 standard) In this example, we use g++-13
(if not, install it)

```g++ --version```

```gcc --version```

**Note:** the path to the g++ compiler must be specified in the build system.

Check if Python is installed (it must be Python 3.10 or higher)
(if not, install)

```python3 --version``` or

```which python3.10```


### №2. We find the directory paths to the Python header files and to the dynamic Python libraries.

The command "which" python3.10 will show the path to the python3.10 interpreter

In the example with Linux (Ubuntu) (which python3.10 will output /usr/bin/python3.10)

This command will output the location of the Python3.10 interpreter binary file (in our case), which is located in the directory where the Python header files and dynamic libraries are stored.

These paths will be necessary when building a C++ program with compilation rules specified

The directory with python header files is located at:
- /usr/include/python3.10

Directory with dynamic libraries:
- /usr/lib/python3.10/config-3.10-x86_64-linux-gnu

The python header files are usually located in the .../include/python3.x 
A directory with dynamic libraries ../lib/python3.x/.. 

**Note:** These paths will be needed when building a C++ program with compilation rules, for example, in a Makefile or in other C++ project assembly systems.

### №3. Installing the fuzzyops library

Python libraries are installed using the command `pip install "library name"`

``pip install fuzzyops`` - will install the fuzzyops library and all the dependencies
it uses

**Note:** If you are not using the anaconda environment, and there are several versions of python on the system, then you need to install the fuzzyops library using a specific pip and in a specific directory. 
Python libraries are installed in the ../site-packages, or ../dist-packages directory.

In our case, such a directory is located at /usr/local/lib/python3.10/dist-packages. This directory is usually located at ../local/lib/python3.10/site-packages./

To install the fuzzyops library there, run
the pip install command with the --target flag.

```pip install fuzzyops --target=/usr/local/lib/python3.10/dist-packages```

pip must also be used in the version of Python that you selected.
You can also view the path to pip using the command

```which pip```

**Note:** In the example, the Ubuntu 22.04 system has a single version of the Python interpreter installed, so the installation was performed using the command
```pip install fuzzyops```

### №4. Using

After installing fuzzyops on the system or in the environment, you can access libraries
from C++. Examples of C++ code for directly calling functions from the library are presented in the corresponding example files.

The directory contains an example Makefile for the atomic assembly and execution of an example file.

To check the examples in C++:

- clone or fork the library or download the cpp directory
- open in the editor, for example (https://github.com/microsoft/vscode)
- install the fuzzyops library in the system or environment, following the ([№1](1-before-installation), №3)
- edit the Makefile using ([№1](#1-before-installation),
 [№2](#2-we-find-the-directory-paths-to-the-python-header-files-and-to-the-dynamic-python-libraries))
- execute the make command inside the cpp directory
```make```
- complete the goal ./ex
```./ex```

Before running the examples, you need to look at the C++ example code (for example, for 
fuzzy_nn_ex.cpp, you will need to replace the path with the path to Iris.csv after downloading)

**Note:** To build your projects with the fuzzyops library, you should follow the Makefile data, or rather, the commands with compilation flags specifying paths to header files and dynamic Python libraries, discussed in
[№2](#2-we-find-the-directory-paths-to-the-python-header-files-and-to-the-dynamic-python-libraries).


















