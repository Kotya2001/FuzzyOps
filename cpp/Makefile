# Имя исполняемого файла
TARGET = ex

# Компилятор C++
CXX = /usr/local/opt/gcc@13/bin/g++-13

# Путь к заголовкам и библиотекам Python
PYTHON_INCLUDE = /usr/local/Cellar/python@3.10/3.10.16/Frameworks/Python.framework/Versions/3.10/include/python3.10
PYTHON_LIB_PATH = /usr/local/Cellar/python@3.10/3.10.16/Frameworks/Python.framework/Versions/3.10/lib
PYTHON_LIB = -lpython3.10

# Флаги компиляции
CXXFLAGS = -Wall -std=c++11 -I$(PYTHON_INCLUDE)

# Список исходных файлов
SOURCES = fuzzy_logic_ex.cpp

# Правило для сборки исполняемого файла
$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES) -L$(PYTHON_LIB_PATH) $(PYTHON_LIB)

# Правило для очистки скомпилированных файлов
clean:
	rm -f $(TARGET)