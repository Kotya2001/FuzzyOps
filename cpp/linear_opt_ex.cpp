#include <Python.h>
#include <iostream>
#include <string>
#include <vector>

// Python Initialization
void initialize_python() { Py_Initialize(); }

// Importing a Python module
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

// Converting a Python list to a NumPy array array to pass them to a function
// The numpy library is installed in dependencies with the fuzzyops library
PyObject *list_to_numpy_array(PyObject *pList) {
    PyObject *pNumpyModule = import_module("numpy");
    if (!pNumpyModule) {
        std::cerr << "Failed to import numpy module." << std::endl;
        return nullptr;
    }

    PyObject *pArrayFunc = PyObject_GetAttrString(pNumpyModule, "array");
    if (!pArrayFunc || !PyCallable_Check(pArrayFunc)) {
        std::cerr << "Failed to get [numpy.array](numpy.array) function." << std::endl;
        Py_DECREF(pNumpyModule);
        return nullptr;
    }

    PyObject *pArray = PyObject_CallFunctionObjArgs(pArrayFunc, pList, nullptr);
    Py_DECREF(pArrayFunc);
    Py_DECREF(pNumpyModule);

    if (!pArray) {
        PyErr_Print();
        std::cerr << "Failed to convert list to numpy array." << std::endl;
        return nullptr;
    }

    return pArray;
}

// A function for performing linear optimization
std::pair<double, std::vector<double>> optimize_linear(const std::vector<std::vector<double>> &A,
                                                       const std::vector<double> &b,
                                                       std::vector<std::vector<double>> &C) {
    // Importing the fuzzyops module
    PyObject *pModule = import_module("fuzzyops.fuzzy_optimization");
    if (!pModule)
        return {0.0, {}};

    // Getting a ref to the LinearOptimization class
    PyObject *pLinearOptClass = PyObject_GetAttrString(pModule, "LinearOptimization");
    if (!pLinearOptClass || !PyCallable_Check(pLinearOptClass)) {
        std::cerr << "Failed to get LinearOptimization class." << std::endl;
        Py_DECREF(pModule);
        return {0.0, {}};
    }

    // Converting the input data to Python format, and then to np.ndarray
    PyObject *pA = PyList_New(A.size());
    for (size_t i = 0; i < A.size(); ++i) {
        PyObject *pRow = PyList_New(A[i].size());
        for (size_t j = 0; j < A[i].size(); ++j) {
            PyList_SetItem(pRow, j, PyFloat_FromDouble(A[i][j]));
        }
        PyList_SetItem(pA, i, pRow);
    }

    PyObject *pA_np = list_to_numpy_array(pA);

    PyObject *pB = PyList_New(b.size());
    for (size_t i = 0; i < b.size(); ++i) {
        PyList_SetItem(pB, i, PyFloat_FromDouble(b[i]));
    }

    PyObject *pB_np = list_to_numpy_array(pB);

    // Creating a 2D array C (criterias)
    PyObject *pC = PyList_New(C.size());
    for (size_t i = 0; i < C.size(); ++i) {
        PyObject *pRow = PyList_New(C[i].size());
        for (size_t j = 0; j < C[i].size(); ++j) {
            PyList_SetItem(pRow, j, PyFloat_FromDouble(C[i][j]));
        }
        PyList_SetItem(pC, i, pRow);
    }

    PyObject *pC_np = list_to_numpy_array(pC);

    Py_DECREF(pA);
    Py_DECREF(pB);
    Py_DECREF(pC);

    // Creating an instance of LinearOptimization
    PyObject *pArgs = PyTuple_Pack(4, pA_np, pB_np, pC_np, PyUnicode_FromString("max"));
    PyObject *pLinearOptInstance = PyObject_CallObject(pLinearOptClass, pArgs);
    Py_DECREF(pArgs);

    if (!pLinearOptInstance) {
        PyErr_Print();
        std::cerr << "Failed to create LinearOptimization instance." << std::endl;
        Py_DECREF(pLinearOptClass);
        Py_DECREF(pModule);
        return {0.0, {}};
    }

    // Calling the solve_cpu method
    PyObject *pSolveMethod = PyObject_GetAttrString(pLinearOptInstance, "solve_cpu");
    if (!pSolveMethod || !PyCallable_Check(pSolveMethod)) {
        std::cerr << "'solve_cpu' method not found or is not callable." << std::endl;
        Py_DECREF(pLinearOptInstance);
        Py_DECREF(pLinearOptClass);
        Py_DECREF(pModule);
        return {0.0, {}};
    }

    PyObject *pResult = PyObject_CallObject(pSolveMethod, NULL);
    if (!pResult) {
        PyErr_Print();
        std::cerr << "Failed to call 'solve_cpu'." << std::endl;
        Py_DECREF(pSolveMethod);
        Py_DECREF(pLinearOptInstance);
        Py_DECREF(pLinearOptClass);
        Py_DECREF(pModule);
        return {0.0, {}};
    }

    // We get the result
    std::vector<double> values;
    double r = PyFloat_AsDouble(PyTuple_GetItem(pResult, 0));
    PyObject *pValuesList = PyTuple_GetItem(pResult, 1);

    // some functions may return np.ndarray,
    // inside Python, this data type has a tolist method,
    // which converts to the standard list data type
    if (PyObject_HasAttrString(pValuesList, "tolist")) {
        // Convert the tensor to a regular Python list
        PyObject *pListMethod = PyObject_GetAttrString(pValuesList, "tolist");
        if (pListMethod) {
            PyObject *pList = PyObject_CallObject(pListMethod, NULL);
            if (pList) {
                // Getting the size of the list
                Py_ssize_t size = PyList_Size(pList);
                // Saving the values to a C++ vector
                for (Py_ssize_t i = 0; i < size; ++i) {
                    PyObject *pItem = PyList_GetItem(pList, i);
                    if (pItem) {
                        double value = PyFloat_AsDouble(pItem);
                        values.push_back(value);
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

    // Freeing up resources
    Py_DECREF(pResult);
    Py_DECREF(pSolveMethod);
    Py_DECREF(pLinearOptInstance);
    Py_DECREF(pLinearOptClass);
    Py_DECREF(pModule);
    Py_DECREF(pA_np);
    Py_DECREF(pB_np);
    Py_DECREF(pC_np);
    Py_DECREF(pValuesList);

    return {r, values}; // We return the result and the vector of values
};

int main() {
    initialize_python();
    // creating the matrix A
    std::vector<std::vector<double>> A = {{2, 3}, {-1, 3}, {2, -1}};
    // creating the matrix B
    std::vector<double> b = {18, 9, 10};
    // C (values for criteria) must be 2D
    std::vector<std::vector<double>> C = {{4, 2}};
    // We get the result
    auto result = optimize_linear(A, b, C);
    std::cout << "Optimal value: " << result.first << std::endl;
    std::cout << "Optimal variables: ";
    for (double value : result.second) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    Py_Finalize();
    return 0;
}