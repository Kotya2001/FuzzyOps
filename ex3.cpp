#include <Python.h>
#include <cstdio>
#include <iostream>
#include <vector>

// #include <torch/torch.h>

std::string format_string(const char *format, int value) {
    char buffer[50]; // Достаточно большой буфер для хранения строки
    snprintf(buffer, sizeof(buffer), format, value);
    return std::string(buffer);
}

void initialize_python() { Py_Initialize(); }

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

// void create_gauss_number(PyObject *pDomain, const char *name, double sigma, double mean) {
//     PyObject *pCreateNumber = PyObject_GetAttrString(pDomain, "create_number");
//     PyObject *pArgs = PyTuple_Pack(
//         3, PyUnicode_FromString("gauss"), PyFloat_FromDouble(sigma), PyFloat_FromDouble(mean));
//     // Создаем именованные аргументы (словарь)
//     PyObject *pKwargs = PyDict_New(); // Новый словарь
//     PyDict_SetItemString(pKwargs, "name", PyUnicode_FromString(name));

//     // Вызываем create_number с позиционными и именованными аргументами
//     PyObject *pResult = PyObject_Call(pCreateNumber, pArgs, pKwargs);
//     if (!pResult) {
//         PyErr_Print();
//         std::cerr << "Failed to call create_number" << std::endl;
//     }

//     // Освобождаем ресурсы
//     Py_DECREF(pArgs);
//     Py_DECREF(pKwargs);
//     Py_DECREF(pCreateNumber);
//     if (pResult) {
//         Py_DECREF(pResult); // Освобождаем результат, если он не NULL
//     }
// }

// void get_f(PyObject *f_num) {
//     PyObject *pStrMethod = PyObject_GetAttrString(f_num, "__str__");
//     if (pStrMethod) {
//         // Вызываем метод __str__()
//         PyObject *pStrRep = PyObject_CallObject(pStrMethod, NULL);
//         if (pStrRep) {
//             // Выводим строковое представление на консоль
//             std::cout << "String representation of f_num " << PyUnicode_AsUTF8(pStrRep)
//                       << std::endl;
//             // Освобождаем ресурсы
//             Py_DECREF(pStrRep);
//         } else {
//             PyErr_Print();
//             std::cerr << "Failed to call __str__ on 'out'" << std::endl;
//         }
//         Py_DECREF(pStrMethod);
//     } else {
//         PyErr_Print();
//         std::cerr << "Failed to get __str__ method from 'out'" << std::endl;
//     }
// };

// PyObject *sum_numbers(PyObject *pDomain) {

//     PyObject *summator = PyObject_GetAttrString(pDomain, "out");
//     if (!summator) {
//         PyErr_Print();
//         std::cerr << "Failed to get summator" << std::endl;
//     }

//     // Начинаем с нулевого значения для итогового результата
//     PyObject *total_sum = summator; // создаем копию summator
//     if (!total_sum) {
//         PyErr_Print();
//         std::cerr << "Failed to initialize total_sum" << std::endl;
//         Py_DECREF(summator);
//         return NULL; // Возвращаем NULL при ошибке
//     }

//     for (int i = 0; i < 15; i++) {

//         std::string nNameStr = format_string("n%d", i);
//         create_gauss_number(pDomain, nNameStr.c_str(), 1.0, i);

//         std::cout << nNameStr << std::endl;

//         PyObject *pAttrib1 = PyObject_GetAttrString(pDomain, nNameStr.c_str());
//         if (!pAttrib1) {
//             PyErr_Print();
//             std::cerr << "Failed to get value on domain: " << nNameStr << std::endl;
//         }
//         PyObject *pSumMethod = PyObject_GetAttrString(summator, "__iadd__");
//         if (pSumMethod) {
//             PyObject *pResult = PyObject_CallFunctionObjArgs(pSumMethod, pAttrib1, NULL);
//             if (!pResult) {
//                 PyErr_Print();
//                 std::cerr << "Failed to add value to out" << std::endl;
//             } else {
//                 // Обновляем итоговую сумму
//                 PyObject *tmp_sum = total_sum; // Сохраняем текущую сумму
//                 total_sum = pResult;           // Обновляем сумму

//                 // Освобождаем старую сумму
//                 Py_DECREF(tmp_sum);
//             }
//             Py_DECREF(pSumMethod); // Освобождаем метод сложения
//         } else {
//             PyErr_Print();
//             std::cerr << "Failed to get summation method" << std::endl;
//         }

//         Py_DECREF(pAttrib1);
//     }
//     return total_sum;
// }

// void get_values(PyObject *pDomain, const char *device) {
//     PyObject *pToMethod = PyObject_GetAttrString(pDomain, "to");
//     if (pToMethod) {
//         // Создаем аргумент для метода 'to'
//         PyObject *pArgs = PyTuple_Pack(1, PyUnicode_FromString(device));
//         PyObject *pResult = PyObject_CallObject(pToMethod, pArgs);
//         if (!pResult) {
//             PyErr_Print();
//             std::cerr << "Failed to call to('cpu' or 'cuda')" << std::endl;
//         }
//         // Освобождаем ресурсы
//         Py_DECREF(pArgs);
//         Py_DECREF(pToMethod);
//         Py_XDECREF(pResult); // Освобождаем pResult, если он не NULL

//     } else {
//         PyErr_Print();
//         std::cerr << "Failed to get method 'to'" << std::endl;
//         return;
//     }

//     // Установка метода 'method' на 'minimax'
//     PyObject *pMethodAttr = PyUnicode_FromString("minimax");
//     if (pMethodAttr) {
//         if (PyObject_SetAttrString(pDomain, "method", pMethodAttr) < 0) {
//             PyErr_Print();
//             std::cerr << "Failed to set method to 'minimax'" << std::endl;
//         }
//         Py_DECREF(pMethodAttr);
//     }

//     PyObject *pOut = PyObject_GetAttrString(pDomain, "out");
//     if (pOut) {
//         PyObject *pValuesAttr = PyObject_GetAttrString(pOut, "values");
//         std::cout << "Values" << std::endl;
//         if (pValuesAttr) {
//             if (PyList_Check(pValuesAttr)) {
//                 Py_ssize_t size = PyList_Size(pValuesAttr);
//                 std::cout << "Values:" << std::endl;
//                 std::cout << size << std::endl;
//                 for (Py_ssize_t i = 0; i < size; ++i) {
//                     PyObject *pValue = PyList_GetItem(pValuesAttr, i);
//                     if (PyFloat_Check(pValue)) {
//                         std::cout << "Value: " << PyFloat_AsDouble(pValue) << std::endl;
//                     }
//                 }
//             }
//             // Не забудьте освободить 'values'
//             Py_DECREF(pValuesAttr);
//         } else {
//             PyErr_Print();
//             std::cerr << "Failed to get 'values' from 'out'" << std::endl;
//         }
//         Py_DECREF(pOut);
//     } else {
//         PyErr_Print();
//         std::cerr << "Failed to get 'out' from 'Domain'" << std::endl;
//     }
// }

// void get_f_num(PyObject *pDomain, const char *name) {
//     PyObject *pOut = PyObject_GetAttrString(pDomain, name);
//     if (pOut) {
//         // Превращаем pOut в строку, вызывая __str__()
//         PyObject *pStrMethod = PyObject_GetAttrString(pOut, "__str__");
//         if (pStrMethod) {
//             // Вызываем метод __str__()
//             PyObject *pStrRep = PyObject_CallObject(pStrMethod, NULL);
//             if (pStrRep) {
//                 // Выводим строковое представление на консоль
//                 std::cout << "String representation of d.out: " << PyUnicode_AsUTF8(pStrRep)
//                           << std::endl;
//                 // Освобождаем ресурсы
//                 Py_DECREF(pStrRep);
//             } else {
//                 PyErr_Print();
//                 std::cerr << "Failed to call __str__ on 'out'" << std::endl;
//             }
//             Py_DECREF(pStrMethod);
//         } else {
//             PyErr_Print();
//             std::cerr << "Failed to get __str__ method from 'out'" << std::endl;
//         }
//         Py_DECREF(pOut);
//     } else {
//         PyErr_Print();
//         std::cerr << "Failed to get 'out' from 'd'" << std::endl;
//     }
// };

int main() {
    initialize_python();
    PyObject *d = create_domain(0.0, 101, "d");
    d->method();
    // create_gauss_number(d, "out", 1.0, 0.0);
    // PyObject *result = sum_numbers(d);
    // get_values(d, "cpu");
    // get_f(result);
    // Py_DECREF(result);
    Py_DECREF(d);
    Py_Finalize();
    return 0;
}