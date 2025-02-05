#include <Python.h>
#include <iostream>

void initialize_python() {
    Py_Initialize();
    // PyRun_SimpleString("import sys");
    // PyRun_SimpleString("sys.path.append('.')"); // Указание на текущую директорию
}

PyObject *import_module(const char *module_name) {
    PyObject *pName = PyUnicode_FromString(module_name);
    PyObject *pModule = PyImport_Import(pName);
    Py_DECREF(pName);
    return pModule;
}

PyObject *create_domain(double min_val, double max_val, double step_val, const char *name) {
    // Импортируем модуль fuzzyops.fuzzy_numbers
    PyObject *pDomainModule = import_module("fuzzyops.fuzzy_numbers");
    PyObject *pDomainClass = PyObject_GetAttrString(pDomainModule, "Domain");

    // Создаем кортеж с параметрами
    PyObject *pArgs = PyTuple_Pack(
        3, PyFloat_FromDouble(min_val), PyFloat_FromDouble(max_val), PyFloat_FromDouble(step_val));

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

void create_triangular_number(PyObject *pDomain, const char *name, double a, double b, double c) {
    PyObject *pCreateNumber = PyObject_GetAttrString(pDomain, "create_number");
    PyObject *pArgs = PyTuple_Pack(4,
                                   PyUnicode_FromString("triangular"),
                                   PyFloat_FromDouble(a),
                                   PyFloat_FromDouble(b),
                                   PyFloat_FromDouble(c));
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

double calculate_final_scores(PyObject *pDomain, const char *name1, const char *name2) {
    PyObject *pCalcFinalScores = import_module("fuzzyops.fan"); // Импортируем модуль fan
    PyObject *pCalcMethod = PyObject_GetAttrString(pCalcFinalScores, "calc_final_scores");

    std::cout << "Impotred" << std::endl;

    PyObject *pAttrib1 = PyObject_GetAttrString(pDomain, name1);
    if (!pAttrib1) {
        PyErr_Print();
        std::cerr << "Failed to get attribute: " << name1 << std::endl;
        return -1;
    }

    PyObject *pAttrib2 = PyObject_GetAttrString(pDomain, name2);
    if (!pAttrib2) {
        PyErr_Print();
        std::cerr << "Failed to get attribute: " << name2 << std::endl;
        Py_DECREF(pAttrib1);
        return -1;
    }

    // PyObject *pArgs = PyTuple_Pack(
    //     2, PyObject_GetAttrString(pDomain, name1), PyObject_GetAttrString(pDomain, name2));
    // PyObject *pResult = PyObject_CallObject(pCalcMethod, pArgs);
    // std::cout << "Called" << std::endl;
    // double score = PyFloat_AsDouble(pResult);

    // Создаем кортеж для передачи в функцию
    PyObject *pArgs = PyTuple_New(2); // Создаем кортеж из 2 элементов
    if (!pArgs) {
        PyErr_Print();
        std::cerr << "Failed to create tuple" << std::endl;
        Py_DECREF(pAttrib1);
        Py_DECREF(pAttrib2);
        return -1;
    }

    // Заполняем кортеж
    PyTuple_SetItem(pArgs, 0, pAttrib1); // Присваиваем pAttrib1 в кортеж
    PyTuple_SetItem(pArgs, 1, pAttrib2); // Присваиваем pAttrib2 в кортеж

    // Вызываем метод
    PyObject *pResult = PyObject_CallObject(pCalcMethod, pArgs);
    if (!pResult) {
        PyErr_Print();
        std::cerr << "Failed to call calc_final_scores" << std::endl;
        Py_DECREF(pArgs);
        Py_DECREF(pCalcMethod);
        return -1;
    }

    // Получаем итоговый балл
    double score = PyFloat_AsDouble(pResult);

    // Освобождаем ресурсы
    Py_DECREF(pResult);
    Py_DECREF(pCalcMethod);
    Py_DECREF(pArgs); // Освобождаем кортеж

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

void find_most_feasible_path(PyObject *graph) {
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

int main() {
    initialize_python();

    PyObject *pScoreDomain = create_domain(0.0, 1.0, 0.01, "scores");

    create_triangular_number(pScoreDomain, "complex_A", 0.4, 0.7, 0.9);
    create_triangular_number(pScoreDomain, "sources_A", 0.4, 0.76, 1);
    double A_score = calculate_final_scores(pScoreDomain, "complex_A", "sources_A");

    // create_triangular_number(pScoreDomain, "complex_A2", 0.5, 0.7, 0.95);
    // create_triangular_number(pScoreDomain, "sources_A2", 0.4, 0.85, 1);
    // double A2_score = calculate_final_scores(pScoreDomain, "complex_A2", "sources_A2");

    // create_triangular_number(pScoreDomain, "complex_B", 0.5, 0.77, 1);
    // create_triangular_number(pScoreDomain, "sources_B", 0.44, 0.89, 1);
    // double B_score = calculate_final_scores(pScoreDomain, "complex_B", "sources_B");

    // create_triangular_number(pScoreDomain, "complex_C", 0.3, 0.6, 0.91);
    // create_triangular_number(pScoreDomain, "sources_C", 0.4, 0.7, 1);
    // double C_score = calculate_final_scores(pScoreDomain, "complex_C", "sources_C");

    // create_triangular_number(pScoreDomain, "complex_D", 0.4, 0.61, 0.81);
    // create_triangular_number(pScoreDomain, "sources_D", 0.43, 0.7, 1);
    // double D_score = calculate_final_scores(pScoreDomain, "complex_D", "sources_D");

    // create_triangular_number(pScoreDomain, "complex_E", 0.5, 0.6, 0.91);
    // create_triangular_number(pScoreDomain, "sources_E", 0.5, 0.8, 1);
    // double E_score = calculate_final_scores(pScoreDomain, "complex_E", "sources_E");

    // PyObject *graph = import_module("fuzzyops.fan"); // Импортируем Graph
    // PyObject *pGraphClass = PyObject_GetAttrString(graph, "Graph");
    // PyObject *pGraphInstance = PyObject_CallObject(pGraphClass, nullptr);

    // add_edge_to_graph(pGraphInstance, "Start", "A", A_score);
    // add_edge_to_graph(pGraphInstance, "Start", "A2", A2_score);
    // add_edge_to_graph(pGraphInstance, "A", "B", std::max(A_score, B_score));
    // add_edge_to_graph(pGraphInstance, "A2", "B", std::max(A2_score, B_score));
    // add_edge_to_graph(pGraphInstance, "B", "C", std::max(C_score, B_score));
    // add_edge_to_graph(pGraphInstance, "C", "D", std::max(C_score, D_score));
    // add_edge_to_graph(pGraphInstance, "C", "E", std::max(C_score, E_score));
    // add_edge_to_graph(pGraphInstance, "D", "End", D_score);
    // add_edge_to_graph(pGraphInstance, "E", "End", E_score);

    // find_most_feasible_path(pGraphInstance);

    // // Освобождаем ресурсы
    // Py_DECREF(pGraphInstance);
    // Py_DECREF(pGraphClass);
    // Py_DECREF(graph);
    Py_DECREF(pScoreDomain);

    // Завершаем Python
    Py_Finalize();
    return 0;
}