#include <Python.h>
#include <iostream>

void initialize_python() { Py_Initialize(); }

PyObject *import_module(const char *module_name) {
    PyObject *pName = PyUnicode_FromString(module_name);
    PyObject *pModule = PyImport_Import(pName);
    Py_DECREF(pName);
    return pModule;
}

PyObject *create_domain(double min_val, double max_val, double step_val, const char *name) {
    // Importing the module fuzzyops.fuzzy_numbers
    PyObject *pDomainModule = import_module("fuzzyops.fuzzy_numbers");
    PyObject *pDomainClass = PyObject_GetAttrString(pDomainModule, "Domain");

    // Creating a tuple with parameters
    PyObject *pArgs = PyTuple_Pack(
        3, PyFloat_FromDouble(min_val), PyFloat_FromDouble(max_val), PyFloat_FromDouble(step_val));

    // Creating a tuple to pass to the constructor Domain
    PyObject *pDomainTuple = PyTuple_Pack(1, pArgs); // The outer tuple
    PyObject *pDomain = PyObject_CallObject(pDomainClass, pDomainTuple);

    // Freeing up resources
    Py_DECREF(pArgs);
    Py_DECREF(pDomainTuple);
    Py_DECREF(pDomainClass);
    Py_DECREF(pDomainModule);

    // Setting the name attribute
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
    // Creating named arguments (dictionary)
    PyObject *pKwargs = PyDict_New(); // New Dictionary
    PyDict_SetItemString(pKwargs, "name", PyUnicode_FromString(name));

    // Calling create_number with positional and named arguments
    PyObject *pResult = PyObject_Call(pCreateNumber, pArgs, pKwargs);
    if (!pResult) {
        PyErr_Print();
        std::cerr << "Failed to call create_number" << std::endl;
    }

    // Freeing up resources
    Py_DECREF(pArgs);
    Py_DECREF(pKwargs);
    Py_DECREF(pCreateNumber);
    if (pResult) {
        Py_DECREF(pResult); // We release the result if it is not NULL.
    }
}

double calculate_final_scores(PyObject *pDomain, const char *name1, const char *name2) {
    PyObject *pCalcFinalScores = import_module("fuzzyops.fan"); // Importing the fan module
    PyObject *pCalcMethod = PyObject_GetAttrString(pCalcFinalScores, "calc_final_scores");

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

    PyObject *pNumList = PyList_New(2);
    if (!pNumList) {
        PyErr_Print();
        std::cerr << "Failed to create list" << std::endl;
        Py_DECREF(pAttrib1);
        Py_DECREF(pAttrib2);
        return -1;
    }

    PyList_SetItem(pNumList, 0, pAttrib1);
    PyList_SetItem(pNumList, 1, pAttrib2);

    PyObject *pArgs = PyTuple_Pack(1, pNumList);

    // Calling the method
    PyObject *pResult = PyObject_CallObject(pCalcMethod, pArgs);
    if (!pResult) {
        PyErr_Print();
        std::cerr << "Failed to call calc_final_scores" << std::endl;
        Py_DECREF(pArgs);
        Py_DECREF(pCalcMethod);
        return -1;
    }

    // Getting the final score
    double score = PyFloat_AsDouble(pResult);

    // Освобождаем ресурсы
    Py_DECREF(pResult);
    Py_DECREF(pCalcMethod);
    Py_DECREF(pNumList); // Releasing the tuple
    Py_DECREF(pArgs);

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
    // Creating a domain
    PyObject *pScoreDomain = create_domain(0.0, 1.0, 0.01, "scores");
    // creating fuzzy numbers of criteria for assigning degrees of feasibility
    // to edges
    create_triangular_number(pScoreDomain, "complex_A", 0.4, 0.7, 0.9);
    create_triangular_number(pScoreDomain, "sources_A", 0.4, 0.76, 1);
    double A_score = calculate_final_scores(pScoreDomain, "complex_A", "sources_A");

    create_triangular_number(pScoreDomain, "complex_A2", 0.5, 0.7, 0.95);
    create_triangular_number(pScoreDomain, "sources_A2", 0.4, 0.85, 1);
    double A2_score = calculate_final_scores(pScoreDomain, "complex_A2", "sources_A2");

    create_triangular_number(pScoreDomain, "complex_B", 0.5, 0.77, 1);
    create_triangular_number(pScoreDomain, "sources_B", 0.44, 0.89, 1);
    double B_score = calculate_final_scores(pScoreDomain, "complex_B", "sources_B");

    create_triangular_number(pScoreDomain, "complex_C", 0.3, 0.6, 0.91);
    create_triangular_number(pScoreDomain, "sources_C", 0.4, 0.7, 1);
    double C_score = calculate_final_scores(pScoreDomain, "complex_C", "sources_C");

    create_triangular_number(pScoreDomain, "complex_D", 0.4, 0.61, 0.81);
    create_triangular_number(pScoreDomain, "sources_D", 0.43, 0.7, 1);
    double D_score = calculate_final_scores(pScoreDomain, "complex_D", "sources_D");

    create_triangular_number(pScoreDomain, "complex_E", 0.5, 0.6, 0.91);
    create_triangular_number(pScoreDomain, "sources_E", 0.5, 0.8, 1);
    double E_score = calculate_final_scores(pScoreDomain, "complex_E", "sources_E");

    // Creating a Graph
    PyObject *graph = import_module("fuzzyops.fan");
    PyObject *pGraphClass = PyObject_GetAttrString(graph, "Graph");
    PyObject *pGraphInstance = PyObject_CallObject(pGraphClass, nullptr);

    // Adding edges to the graph
    add_edge_to_graph(pGraphInstance, "Start", "A", A_score);
    add_edge_to_graph(pGraphInstance, "Start", "A2", A2_score);
    add_edge_to_graph(pGraphInstance, "A", "B", std::max(A_score, B_score));
    add_edge_to_graph(pGraphInstance, "A2", "B", std::max(A2_score, B_score));
    add_edge_to_graph(pGraphInstance, "B", "C", std::max(C_score, B_score));
    add_edge_to_graph(pGraphInstance, "C", "D", std::max(C_score, D_score));
    add_edge_to_graph(pGraphInstance, "C", "E", std::max(C_score, E_score));
    add_edge_to_graph(pGraphInstance, "D", "End", D_score);
    add_edge_to_graph(pGraphInstance, "E", "End", E_score);

    find_most_feasible_path(pGraphInstance);

    // Freeing up resources
    Py_DECREF(pGraphInstance);
    Py_DECREF(pGraphClass);
    Py_DECREF(graph);
    Py_DECREF(pScoreDomain);

    // Completing Python
    Py_Finalize();
    return 0;
}