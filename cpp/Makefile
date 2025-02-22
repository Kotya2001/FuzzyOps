# Имя исполняемого файла
TARGET = ex

# Путь к компилятору C++ (Замените на ващ)
CXX = /usr/local/opt/gcc@13/bin/g++-13

# Путь к заголовочным файлам Python(необходимо заменить на ваши пути)
PYTHON_INCLUDE = /usr/local/Cellar/python@3.10/3.10.16/Frameworks/Python.framework/Versions/3.10/include/python3.10
# Путь к динамическим библиотекам Python(необходимо заменить на ваши пути)
PYTHON_LIB_PATH = /usr/local/Cellar/python@3.10/3.10.16/Frameworks/Python.framework/Versions/3.10/lib
# Данная переменная меняется в зависимости от версии python, 
# которую вы используете (для 3.10 будет -lpython3.10)
PYTHON_LIB = -lpython3.10

# Флаги компиляции
CXXFLAGS = -Wall -std=c++11 -I$(PYTHON_INCLUDE)

# Список исходных файлов (путь к конкретному примеру относительно директории сpp)
SOURCES = fuzzy_nn_ex.cpp

# Правило для сборки исполняемого файла
$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES) -L$(PYTHON_LIB_PATH) $(PYTHON_LIB)

# Правило для очистки скомпилированных файлов
clean:
	rm -f $(TARGET)