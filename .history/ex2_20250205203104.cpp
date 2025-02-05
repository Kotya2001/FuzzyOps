#include <Python.h>
#include <iostream>

PyObject *import_module(const char *module_name) {
    PyObject *pName = PyUnicode_FromString(module_name);
    PyObject *pModule = PyImport_Import(pName);
    Py_DECREF(pName);
    return pModule;
}

PyObject *create_domain(double min_val, double max_val, double step_val, const char *name) {
    PyObject *pDomainModule = import_module("fuzzyops.fuzzy_numbers");
    PyObject *pDomainClass = PyObject_GetAttrString(pDomainModule, "Domain");

    PyObject *pArgs = PyTuple_Pack(
        3, PyFloat_FromDouble(min_val), PyFloat_FromDouble(max_val), PyFloat_FromDouble(step_val));
    PyObject *pDomain = PyObject_CallObject(pDomainClass, pArgs);
    Py_DECREF(pArgs);
    Py_DECREF(pDomainClass);
    Py_DECREF(pDomainModule);

    if (pDomain) {
        PyObject_SetAttrString(pDomain, "name", PyUnicode_FromString(name));
    } else {
        PyErr_Print();
    }

    return pDomain;
}

void create_triangular_number(PyObject *pDomain, const char *name, double a, double b, double c) {
    PyObject *pCreateNumber = PyObject_GetAttrString(pDomain, "create_number");
    PyObject *pArgs = PyTuple_Pack(4,
                                   PyUnicode_FromString("triangular"),
                                   PyFloat_FromDouble(a),
                                   PyFloat_FromDouble(b),
                                   PyFloat_FromDouble(c));
    PyObject_CallObject(pCreateNumber, pArgs);
    Py_DECREF(pArgs);
    Py_DECREF(pCreateNumber);
}

double calculate_final_scores(PyObject *pDomain, const char *name1, const char *name2) {
    PyObject *pCalcFinalScores = import_module("fuzzyops.fan"); // Импортируем модуль fan
    PyObject *pCalcMethod = PyObject_GetAttrString(pCalcFinalScores, "calc_final_scores");

    PyObject *pArgs = PyTuple_Pack(
        2, PyObject_GetAttrString(pDomain, name1), PyObject_GetAttrString(pDomain, name2));
    PyObject *pResult = PyObject_CallObject(pCalcMethod, pArgs);
    double score = PyFloat_AsDouble(pResult);

    // Освобождаем ресурсы
    Py_DECREF(pArgs);
    Py_DECREF(pResult);
    Py_DECREF(pCalcMethod);
    Py_DECREF(pCalcFinalScores);

    return score;
}
void add_edge_to_graph(PyObject *graph,
                       const char *from_node,
                       const char *to_node,
                       double weight) {
    PyObject *pAddEdgeMethod = PyObject_GetAttrString(graph, "add_edge");
    PyObject *pArgs = PyTuple_Pack(3,
                                   PyUnicode_FromString(from_node),
                                   PyUnicode_FromString(to_node),
                                   PyFloat_FromDouble(weight));
    PyObject_CallObject(pAddEdgeMethod, pArgs);
    Py_DECREF(pArgs);
    Py_DECREF(pAddEdgeMethod);
}

... void find_most_feasible_path(PyObject *graph) {
    PyObject *pFindPathMethod = PyObject_GetAttrString(graph, "find_most_feasible_path");
    PyObject *pArgs = PyTuple_Pack(2, PyUnicode_FromString("Start"), PyUnicode_FromString("End"));
    PyObject *pPath = PyObject_CallObject(pFindPathMethod, pArgs);

    if (pPath) {
        std::cout << "Most feasible path: ";
        PyObject_Print(pPath, stdout, 0);
        std::cout << std::endl;

        // Вычисление фуззiness
        PyObject *pCalcFuzziness = PyObject_GetAttrString(graph, "calculate_path_fuzziness");
        PyObject *pFuzziness = PyObject_CallObject(pCalcFuzziness, PyTuple_Pack(1, pPath));

        std::cout << "Feasibility: " << PyFloat_AsDouble(pFuzziness) << std::endl;

        Py_DECREF(pFuzziness);
        Py_DECREF(pCalcFuzziness);
        Py_DECREF(pPath);
    } else {
        std::cout << "No path found." << std::endl;
    }

    Py_DECREF(pArgs);
    Py_DECREF(pFindPathMethod);
}
