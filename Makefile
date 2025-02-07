# # Имя исполняемого файла
# TARGET = python_graph_example

# # Компилятор C++
# CXX = /usr/local/opt/gcc@13/bin/g++-13

# # Флаги компиляции
# CXXFLAGS = -Wall -std=c++11 -I/usr/local/Cellar/python@3.10/3.10.16/Frameworks/Python.framework/Versions/3.10/include/python3.10 -L/usr/local/Cellar/python@3.10/3.10.16/Frameworks/Python.framework/Versions/3.10/lib -lpython3.10

# # Список исходных файлов
# SOURCES = ex.cpp

# # Правило для сборки исполняемого файла
# $(TARGET): $(SOURCES)
# 	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

# # Правило для очистки скомпилированных файлов
# clean:
# 	rm -f $(TARGET)

# Имя исполняемого файла
TARGET = python_graph_example

# Компилятор C++
CXX = /usr/local/opt/gcc@13/bin/g++-13

# Путь к заголовкам и библиотекам Python
PYTHON_INCLUDE = /usr/local/Cellar/python@3.10/3.10.16/Frameworks/Python.framework/Versions/3.10/include/python3.10
PYTHON_LIB_PATH = /usr/local/Cellar/python@3.10/3.10.16/Frameworks/Python.framework/Versions/3.10/lib
PYTHON_LIB = -lpython3.10

# Флаги компиляции
CXXFLAGS = -Wall -std=c++11 -I$(PYTHON_INCLUDE)

# Список исходных файлов
SOURCES = ex3.cpp

# Правило для сборки исполняемого файла
$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES) -L$(PYTHON_LIB_PATH) $(PYTHON_LIB)

# Правило для очистки скомпилированных файлов
clean:
	rm -f $(TARGET)