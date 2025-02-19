#include <Python.h>
#include <cstdio>
#include <iostream>
#include <vector>

std::string format_string(const char *format, int value) {
    char buffer[50]; // Достаточно большой буфер для хранения строки
    snprintf(buffer, sizeof(buffer), format, value);
    return std::string(buffer);
}

void initialize_python() { Py_Initialize(); }

// Функция для импорта модуля из бибилиотеки fuzzyops
PyObject *import_module(const char *module_name) {
    PyObject *pName = PyUnicode_FromString(module_name);
    PyObject *pModule = PyImport_Import(pName);
    Py_DECREF(pName);
    if (!pModule) {
        PyErr_Print();
        std::cerr << "Failed to load module" << std::endl;
        Py_DECREF(pModule);
        return {};
    }
    return pModule;
}

// Функция для импорта модуля из бибилиотеки fuzzyops
PyObject *create_domain(double min_val, double max_val, const char *name) {
    // Импортируем модуль fuzzyops.fuzzy_numbers
    PyObject *pDomainModule = import_module("fuzzyops.fuzzy_numbers");
    PyObject *pDomainClass = PyObject_GetAttrString(pDomainModule, "Domain");

    // Создаем кортеж с параметрами
    PyObject *pArgs = PyTuple_Pack(2, PyFloat_FromDouble(min_val), PyFloat_FromDouble(max_val));

    // Создаем кортеж для передачи в конструктор Domain
    PyObject *pDomainTuple = PyTuple_Pack(1, pArgs); // Внешний кортеж
    PyObject *pDomain = PyObject_CallObject(pDomainClass, pDomainTuple);

    // Освобождаем ресурсы
    Py_DECREF(pArgs);
    Py_DECREF(pDomainTuple);
    Py_DECREF(pDomainClass);
    Py_DECREF(pDomainModule);

    // Устанавливаем атрибут name
    if (pDomain) {
        PyObject_SetAttrString(pDomain, "name", PyUnicode_FromString(name));
    } else {
        PyErr_Print();
    }

    return pDomain;
}

// Функция для создания гауссового числа
void create_gauss_number(PyObject *pDomain, const char *name, double sigma, double mean) {
    PyObject *pCreateNumber = PyObject_GetAttrString(pDomain, "create_number");
    PyObject *pArgs = PyTuple_Pack(
        3, PyUnicode_FromString("gauss"), PyFloat_FromDouble(sigma), PyFloat_FromDouble(mean));

    // Создаем именованные аргументы (словарь)
    PyObject *pKwargs = PyDict_New(); // Новый словарь
    PyDict_SetItemString(pKwargs, "name", PyUnicode_FromString(name));
    // Вызываем create_number с позиционными и именованными аргументами
    PyObject *pResult = PyObject_Call(pCreateNumber, pArgs, pKwargs);
    if (!pResult) {
        PyErr_Print();
        std::cerr << "Failed to call create_number" << std::endl;
    }

    // Освобождаем ресурсы
    Py_DECREF(pArgs);
    Py_DECREF(pKwargs);
    Py_DECREF(pCreateNumber);
    if (pResult) {
        Py_DECREF(pResult); // Освобождаем результат, если он не NULL
    }
}

// Функция для суммирования нечетких чисел и получения итогового нечеткого числа
PyObject *sum_numbers(PyObject *pDomain) {

    PyObject *summator = PyObject_GetAttrString(pDomain, "out");
    if (!summator) {
        PyErr_Print();
        std::cerr << "Failed to get summator" << std::endl;
    }

    // Начинаем с нулевого значения для итогового результата
    PyObject *total_sum = summator; // создаем копию summator
    if (!total_sum) {
        PyErr_Print();
        std::cerr << "Failed to initialize total_sum" << std::endl;
        Py_DECREF(summator);
        return NULL; // Возвращаем NULL при ошибке
    }

    for (int i = 0; i < 50; i++) {

        std::string nNameStr = format_string("n%d", i);
        create_gauss_number(pDomain, nNameStr.c_str(), 1.0, i);

        PyObject *pAttrib1 = PyObject_GetAttrString(pDomain, nNameStr.c_str());
        if (!pAttrib1) {
            PyErr_Print();
            std::cerr << "Failed to get value on domain: " << nNameStr << std::endl;
        }
        PyObject *pSumMethod = PyObject_GetAttrString(summator, "__iadd__");
        if (pSumMethod) {
            PyObject *pResult = PyObject_CallFunctionObjArgs(pSumMethod, pAttrib1, NULL);
            if (!pResult) {
                PyErr_Print();
                std::cerr << "Failed to add value to out" << std::endl;
            } else {
                // Обновляем итоговую сумму
                PyObject *tmp_sum = total_sum; // Сохраняем текущую сумму
                total_sum = pResult;           // Обновляем сумму

                // Освобождаем старую сумму
                Py_DECREF(tmp_sum);
            }
            Py_DECREF(pSumMethod); // Освобождаем метод сложения
        } else {
            PyErr_Print();
            std::cerr << "Failed to get summation method" << std::endl;
        }

        Py_DECREF(pAttrib1);
    }
    return total_sum;
}
// Функция для получения степеней уверенности результата
std::vector<double> print_values(PyObject *fNum, const char *device) {
    std::vector<double> result;

    PyObject *pDomain = PyObject_GetAttrString(fNum, "domain");
    if (!pDomain) {
        PyErr_Print();
        return {}; // Возвращаем пустой vector при ошибке
    }

    // Вызываем метод .to() с аргументом 'cpu'
    PyObject *pToMethod = PyObject_GetAttrString(pDomain, "to");
    if (pToMethod) {
        PyObject *pDevice = PyUnicode_FromString(device);
        PyObject *pOutDevice = PyObject_CallFunctionObjArgs(pToMethod, pDevice, NULL);
        Py_DECREF(pDevice);
        Py_DECREF(pToMethod);

        if (!pOutDevice) {
            PyErr_Print();
            return {};
        }

        // Получаем значения из тензора
        PyObject *pValues = PyObject_GetAttrString(fNum, "values");
        if (pValues) {
            // Проверяем, является ли объект тензором
            if (PyObject_HasAttrString(pValues, "tolist")) {
                // Преобразуем тензор в обычный список Python
                PyObject *pListMethod = PyObject_GetAttrString(pValues, "tolist");
                if (pListMethod) {
                    PyObject *pList = PyObject_CallObject(pListMethod, NULL);
                    if (pList) {
                        // Получаем размер списка
                        Py_ssize_t size = PyList_Size(pList);
                        // Сохраняем значения в вектор C++
                        result.reserve(size);
                        for (Py_ssize_t i = 0; i < size; ++i) {
                            PyObject *pItem = PyList_GetItem(pList, i);
                            if (pItem) {
                                double value = PyFloat_AsDouble(pItem);
                                result.push_back(value);
                            }
                        }
                        Py_DECREF(pList);
                    } else {
                        PyErr_Print();
                    }
                    Py_DECREF(pListMethod);
                }
            } else {
                PyErr_SetString(PyExc_TypeError, "Expected values to be a tensor.");
                PyErr_Print();
            }

            Py_DECREF(pValues);
        } else {
            PyErr_Print();
        }

        Py_DECREF(pOutDevice);
    } else {
        PyErr_Print();
    }
    return result;
}
// Функция для получения дефаззифицированного значения нечеткого числа
double print_f_num(PyObject *f_num) {
    double result = 0.0;

    // Получаем метод __float__
    PyObject *pFloatMethod = PyObject_GetAttrString(f_num, "__float__");
    if (pFloatMethod) {
        // Вызываем метод __float__()
        PyObject *pFloatValue = PyObject_CallObject(pFloatMethod, NULL);
        if (pFloatValue) {
            // Преобразуем результат в double
            result = PyFloat_AsDouble(pFloatValue);
            Py_DECREF(pFloatValue);
        } else {
            PyErr_Print();
            std::cerr << "Failed to call __float__" << std::endl;
        }
        Py_DECREF(pFloatMethod);
    } else {
        PyErr_Print();
        std::cerr << "Failed to get __float__ method" << std::endl;
    }

    PyObject *pStrMethod = PyObject_GetAttrString(f_num, "__str__");
    if (pStrMethod) {
        // Вызываем метод __str__()
        PyObject *pStrRep = PyObject_CallObject(pStrMethod, NULL);
        if (pStrRep) {
            // Выводим строковое представление на консоль
            std::cout << "String representation of f_num " << PyUnicode_AsUTF8(pStrRep)
                      << std::endl;
            // Освобождаем ресурсы
            Py_DECREF(pStrRep);
        } else {
            PyErr_Print();
            std::cerr << "Failed to call __str__" << std::endl;
        }
        Py_DECREF(pStrMethod);
    } else {
        PyErr_Print();
        std::cerr << "Failed to get __str__ method" << std::endl;
    }
    return result;
};

int main() {
    initialize_python();
    // Создание Домена
    PyObject *pDomain = create_domain(0, 101, "d");
    // Создание нечеткого числа
    create_gauss_number(pDomain, "out", 1.0, 0.0);
    // Получение результата суммирования
    PyObject *result = sum_numbers(pDomain);

    // Получение представления нечеткого числа из Python
    // (В Python возвращается дефаззифицированное значение)
    double f_num_value = print_f_num(result);
    std::cout << "Value of f_num: " << f_num_value << std::endl;

    // Получение степеней уверенности у результата суммы
    // :TODO Проверить у Влада на компе с ГПУ
    std::vector<double> values = print_values(result, "cpu");
    std::cout << "Values: ";
    for (double value : values) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    if (pDomain) {
        Py_DECREF(pDomain);
    }
    Py_DECREF(result);

    Py_Finalize();
}