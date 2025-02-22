#include <Python.h>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

void initialize_python() { Py_Initialize(); }

// Функция для импорта модуля из библиотеки fuzzyops
PyObject *import_module(const char *module_name) {
    PyObject *pName = PyUnicode_FromString(module_name);
    PyObject *pModule = PyImport_Import(pName);
    Py_DECREF(pName);
    if (!pModule) {
        PyErr_Print();
        std::cerr << "Failed to load module: " << module_name << std::endl;
        return nullptr;
    }
    return pModule;
}

// Функция для импорта модуля из бибилиотеки fuzzyops
PyObject *create_domain(double min_val, double max_val, double step, const char *name) {
    // Импортируем модуль fuzzyops.fuzzy_numbers
    PyObject *pDomainModule = import_module("fuzzyops.fuzzy_numbers");
    PyObject *pDomainClass = PyObject_GetAttrString(pDomainModule, "Domain");

    // Создаем кортеж с параметрами
    PyObject *pArgs = PyTuple_Pack(
        3, PyFloat_FromDouble(min_val), PyFloat_FromDouble(max_val), PyFloat_FromDouble(step));

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

// Функция для создания нечеткого числа (трапецеидального)
void create_trapezoidal_number(
    PyObject *pDomain, const char *name, double a, double b, double c, double d) {
    PyObject *pCreateNumber = PyObject_GetAttrString(pDomain, "create_number");
    PyObject *pArgs = PyTuple_Pack(5,
                                   PyUnicode_FromString("trapezoidal"),
                                   PyFloat_FromDouble(a),
                                   PyFloat_FromDouble(b),
                                   PyFloat_FromDouble(c),
                                   PyFloat_FromDouble(d));

    // Создание именованных аргументов
    PyObject *pKwargs = PyDict_New();
    PyDict_SetItemString(pKwargs, "name", PyUnicode_FromString(name));

    PyObject *pResult = PyObject_Call(pCreateNumber, pArgs, pKwargs);
    if (!pResult) {
        PyErr_Print();
        std::cerr << "Failed to call create_number" << std::endl;
    }

    Py_DECREF(pArgs);
    Py_DECREF(pKwargs);
    if (pResult) {
        Py_DECREF(pResult); // Освобождаем результат, если он не NULL
    }
}

void check(PyObject *pDomain) {
    if (PyObject_HasAttrString(pDomain, "get")) {
        // Преобразуем тензор в обычный список Python
        PyObject *pListMethod = PyObject_GetAttrString(pDomain, "get");
        if (pListMethod) {
            std::cout << "YES3" << std::endl;
        }
        Py_DECREF(pListMethod);
    } else {
        PyErr_SetString(PyExc_TypeError, "get not found");
        PyErr_Print();
    }
}

// Функция для создания и вычисления вывода нечеткой логики
PyObject *compute_fuzzy_inference(int age) {
    // Импортируем нужные модули
    PyObject *pFuzzyModule = import_module("fuzzyops.fuzzy_logic");
    if (!pFuzzyModule)
        return nullptr;

    // Создаем домены
    PyObject *pAgeDomain = create_domain(0, 100, 1, "age");
    PyObject *pAccidentDomain = create_domain(0, 1, 0.1, "accident");

    // Создаем нечеткие числа для домена возраста
    create_trapezoidal_number(pAgeDomain, "young", -1, 0, 20, 30);
    create_trapezoidal_number(pAgeDomain, "middle", 20, 30, 50, 60);
    create_trapezoidal_number(pAgeDomain, "old", 50, 60, 100, 100);

    // Создаем нечеткие числа для домена аварий
    create_trapezoidal_number(pAccidentDomain, "low", -0.1, 0.0, 0.1, 0.2);
    create_trapezoidal_number(pAccidentDomain, "medium", 0.1, 0.2, 0.7, 0.8);

    create_trapezoidal_number(pAccidentDomain, "high", 0.7, 0.8, 0.9, 1.0);

    // Создаем правила нечеткой логики
    PyObject *pBaseRule = PyObject_GetAttrString(pFuzzyModule, "BaseRule");
    PyObject *rules = PyList_New(0);

    // Добавляем правила
    PyObject *rule1
        = PyObject_CallObject(pBaseRule,
                              PyTuple_Pack(2,
                                           Py_BuildValue("[(s,s)]", "age", "young"),
                                           Py_BuildValue("(s,s)", "accident", "high")));

    PyList_Append(rules, rule1);

    PyObject *rule2
        = PyObject_CallObject(pBaseRule,
                              PyTuple_Pack(2,
                                           Py_BuildValue("[(s,s)]", "age", "middle"),
                                           Py_BuildValue("(s,s)", "accident", "medium")));

    PyList_Append(rules, rule2);

    PyObject *rule3
        = PyObject_CallObject(pBaseRule,
                              PyTuple_Pack(2,
                                           Py_BuildValue("[(s,s)]", "age", "old"),
                                           Py_BuildValue("(s,s)", "accident", "high")));

    PyList_Append(rules, rule3);

    // Создаем экземпляр FuzzyInference, используем Py_BuildValue,
    // чтобы создать dict в Python (хэш-таблица структура данных)

    // Создаем экземпляр FuzzyInference
    PyObject *pFuzzyInferenceClass = PyObject_GetAttrString(pFuzzyModule, "FuzzyInference");
    PyObject *fuzzyInference = PyObject_CallObject(
        pFuzzyInferenceClass,
        PyTuple_Pack(2,
                     Py_BuildValue("{s:O, s:O}", "age", pAgeDomain, "accident", pAccidentDomain),
                     rules));

    // Выполняем вычисление, возвращается объект PyObject *,
    // в Python класс FuzzyNumbers, можно дефаззифицаировать значение,
    // так как данный алгоритм нечеткого логического вывода предполагает
    // дефаззифицированное значение в качестве результата
    PyObject *pResult
        = PyObject_CallMethod(fuzzyInference, "compute", "O", Py_BuildValue("{s:i}", "age", age));

    // Освобождаем ресурсы
    // Py_DECREF(pResult);
    Py_DECREF(fuzzyInference);
    Py_DECREF(pAgeDomain);
    Py_DECREF(pAccidentDomain);
    Py_DECREF(pBaseRule);
    Py_DECREF(pFuzzyModule);

    return pResult;
}

// Функция для получения дефаззифицированного значения нечеткого числа
double get_defuzz_value(PyObject *f_num) {
    double result = 0.0;
    // Получаем метод __float__
    PyObject *pFloatMethod = PyObject_GetAttrString(f_num, "defuzz");
    if (pFloatMethod) {
        // Вызываем метод __float__()
        PyObject *pFloatValue = PyObject_CallObject(pFloatMethod, NULL);
        if (pFloatValue) {
            std::cout << PyFloat_AsDouble(pFloatValue) << std::endl;
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
    return result;
};

int main() {
    initialize_python();

    int age = 25;
    PyObject *pResult = compute_fuzzy_inference(age);

    if (pResult) {
        PyObject *fNum = PyDict_GetItemString(pResult, "accident");
        double result = get_defuzz_value(fNum);
        std::cout << "Computed accident risk for age " << age << ": " << result << std::endl;
    } else {
        PyErr_Print();
        std::cerr << "Failed to compute fuzzy inference." << std::endl;
        return 0;
    }
    Py_DECREF(pResult);

    Py_Finalize();
    return 0;
}