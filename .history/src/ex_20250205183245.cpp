#include <Python.h>
#include <iostream>

int main() {
    // Инициализация интерпретатора Python
    Py_Initialize();

    // Импортируем модуль fan
    PyObject *pName = PyUnicode_FromString("src.fuzzyops.fan.fan");
    PyObject *pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != nullptr) {
        // Получаем класс Graph
        PyObject *pClass = PyObject_GetAttrString(pModule, "Graph");
        PyObject *pGraphInstance = PyObject_CallObject(pClass, nullptr);

        // Добавляем рёбра в граф
        PyObject *pAddEdgeMethod = PyObject_GetAttrString(pGraphInstance, "add_edge");

        // Добавляем ребро A -> B
        PyObject *pArgs1 = PyTuple_Pack(
            3, PyUnicode_FromString("A"), PyUnicode_FromString("B"), PyFloat_FromDouble(0.8));
        PyObject_CallObject(pAddEdgeMethod, pArgs1);
        Py_DECREF(pArgs1);

        // Добавляем ребро B -> C
        PyObject *pArgs2 = PyTuple_Pack(
            3, PyUnicode_FromString("B"), PyUnicode_FromString("C"), PyFloat_FromDouble(0.9));
        PyObject_CallObject(pAddEdgeMethod, pArgs2);
        Py_DECREF(pArgs2);

        // Добавляем ребро A -> D
        PyObject *pArgs3 = PyTuple_Pack(
            3, PyUnicode_FromString("A"), PyUnicode_FromString("D"), PyFloat_FromDouble(0.7));
        PyObject_CallObject(pAddEdgeMethod, pArgs3);
        Py_DECREF(pArgs3);

        // Добавляем ребро D -> C
        PyObject *pArgs4 = PyTuple_Pack(
            3, PyUnicode_FromString("D"), PyUnicode_FromString("C"), PyFloat_FromDouble(0.85));
        PyObject_CallObject(pAddEdgeMethod, pArgs4);
        Py_DECREF(pArgs4);

        // Освобождаем ресурсы
        Py_DECREF(pAddEdgeMethod);

        // Ищем наилучший путь
        PyObject *pMostFeasiblePathMethod
            = PyObject_GetAttrString(pGraphInstance, "find_most_feasible_path");

        PyObject *pArgs5 = PyTuple_Pack(2, PyUnicode_FromString("A"), PyUnicode_FromString("C"));
        PyObject *pMostFeasiblePath = PyObject_CallObject(pMostFeasiblePathMethod, pArgs5);
        Py_DECREF(pArgs5);
        Py_DECREF(pMostFeasiblePathMethod);

        // Проверяем результаты
        if (pMostFeasiblePath != nullptr) {
            std::cout << "Most feasible path from A to C: ";

            // Печатаем путь
            PyObject *pPathStr = PyObject_Str(pMostFeasiblePath);
            const char *path_cstr = PyUnicode_AsUTF8(pPathStr);
            std::cout << path_cstr << std::endl;
            Py_DECREF(pPathStr);

            // Получаем нечеткость пути
            PyObject *pCalculatePathFuzzinessMethod
                = PyObject_GetAttrString(pGraphInstance, "calculate_path_fuzziness");
            PyObject *pFeasibility = PyObject_CallObject(pCalculatePathFuzzinessMethod,
                                                         PyTuple_Pack(1, pMostFeasiblePath));
            Py_DECREF(pCalculatePathFuzzinessMethod);

            double feasibility = PyFloat_AsDouble(pFeasibility);
            std::cout << "Feasibility: " << feasibility << std::endl;
            Py_DECREF(pFeasibility);
        } else {
            std::cout << "No path found." << std::endl;
        }
        Py_DECREF(pMostFeasiblePath);

        // Выполняем макроалгоритм для нахождения наилучшей альтернативы
        PyObject *pBestAlternativeMethod
            = PyObject_GetAttrString(pGraphInstance, "macro_algorithm_for_best_alternative");

        PyObject *pBestAlternative = PyObject_CallObject(pBestAlternativeMethod, nullptr);
        Py_DECREF(pBestAlternativeMethod);

        if (pBestAlternative != nullptr) {
            // Предполагаем, что наилучшая альтернатива возвращается как кортеж (альтернатива,
            // максимальная нечеткость)
            PyObject *pBestAlternativeStr = PyObject_Str(PyTuple_GetItem(pBestAlternative, 0));
            PyObject *pMaxFeasibility = PyTuple_GetItem(pBestAlternative, 1);
            double max_feasibility = PyFloat_AsDouble(pMaxFeasibility);

            const char *best_alternative_cstr = PyUnicode_AsUTF8(pBestAlternativeStr);
            std::cout << "Best alternative: " << best_alternative_cstr << std::endl;
            std::cout << "Max Feasibility: " << max_feasibility << std::endl;

            Py_DECREF(pBestAlternativeStr);
        } else {
            std::cout << "No alternatives found." << std::endl;
        }

        // Освобождаем ресурсы
        Py_DECREF(pBestAlternative);
        Py_DECREF(pGraphInstance);
        Py_DECREF(pClass);
        Py_DECREF(pModule);
    } else {
        PyErr_Print();
        std::cerr << "Failed to load module\n";
    }

    Py_Finalize();
    return 0;
}