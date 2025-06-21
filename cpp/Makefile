# The name of the executable file
TARGET = ex

# The path to the C++ compiler (Replace it with yours)
CXX = /usr/local/opt/gcc@13/bin/g++-13

# Path to the C++ compiler (Replace with your own)
CXX = /usr/bin/g++

# # The path to the Python header files (must be replaced with your paths)
# PYTHON_INCLUDE = /usr/local/Cellar/python@3.10/3.10.16/Frameworks/Python.framework/Versions/3.10/include/python3.10
# # Path to Python dynamic libraries (must be replaced with your paths)
# PYTHON_LIB_PATH = /usr/local/Cellar/python@3.10/3.10.16/Frameworks/Python.framework/Versions/3.10/lib
# # This variable varies depending on the python version.,
# # which you are using (for 3.10 it will be -lpython3.10)
# PYTHON_LIB = -lpython3.10

# The path to the Python header files (must be replaced with your paths)
PYTHON_INCLUDE = /usr/include/python3.10
# Path to Python dynamic libraries (must be replaced with your paths)
PYTHON_LIB_PATH = /usr/lib/python3.10/config-3.10-x86_64-linux-gnu
# This variable varies depending on the python version.,
# # which you are using (for 3.10 it will be -lpython3.10)
PYTHON_LIB = -lpython3.10

# Compilation flags
CXXFLAGS = -Wall -std=c++11 -I$(PYTHON_INCLUDE)

# A list of source files (the path to a specific example relative to the cpp directory that we want to compile)
SOURCES = fuzzy_nn_ex.cpp

# The rule for building an executable file
$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES) -L$(PYTHON_LIB_PATH) $(PYTHON_LIB)

# A rule for cleaning compiled files
clean:
	rm -f $(TARGET)