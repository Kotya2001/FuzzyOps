#include <Python.h>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

void initialize_python() { Py_Initialize(); }

// Function for importing a module from the fuzzyops library
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

// Function for creating a domain
PyObject *create_domain(double min_val, double max_val, double step, const char *name) {
    // Importing the module fuzzyops.fuzzy_numbers
    PyObject *pDomainModule = import_module("fuzzyops.fuzzy_numbers");
    PyObject *pDomainClass = PyObject_GetAttrString(pDomainModule, "Domain");

    // Creating a tuple with parameters
    PyObject *pArgs = PyTuple_Pack(
        3, PyFloat_FromDouble(min_val), PyFloat_FromDouble(max_val), PyFloat_FromDouble(step));

    // Creating a tuple to pass to the Domain constructor
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

// Function for creating a fuzzy number (trapezoidal)
void create_trapezoidal_number(
    PyObject *pDomain, const char *name, double a, double b, double c, double d) {
    PyObject *pCreateNumber = PyObject_GetAttrString(pDomain, "create_number");
    PyObject *pArgs = PyTuple_Pack(5,
                                   PyUnicode_FromString("trapezoidal"),
                                   PyFloat_FromDouble(a),
                                   PyFloat_FromDouble(b),
                                   PyFloat_FromDouble(c),
                                   PyFloat_FromDouble(d));

    // Creating named arguments
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
        Py_DECREF(pResult); // We release the result if it is not NULL.
    }
}

void check(PyObject *pDomain) {
    if (PyObject_HasAttrString(pDomain, "get")) {
        // Converting a tensor to a regular Python list
        PyObject *pListMethod = PyObject_GetAttrString(pDomain, "get");
        
        Py_DECREF(pListMethod);
    } else {
        PyErr_SetString(PyExc_TypeError, "get not found");
        PyErr_Print();
    }
}

// A function for creating and calculating fuzzy logic output
PyObject *compute_fuzzy_inference(int age) {
    // Importing the necessary modules
    PyObject *pFuzzyModule = import_module("fuzzyops.fuzzy_logic");
    if (!pFuzzyModule)
        return nullptr;

    // Creating domains
    PyObject *pAgeDomain = create_domain(0, 100, 1, "age");
    PyObject *pAccidentDomain = create_domain(0, 1, 0.1, "accident");

    // Creating fuzzy numbers for the age domain
    create_trapezoidal_number(pAgeDomain, "young", -1, 0, 20, 30);
    create_trapezoidal_number(pAgeDomain, "middle", 20, 30, 50, 60);
    create_trapezoidal_number(pAgeDomain, "old", 50, 60, 100, 100);

    // Creating fuzzy numbers for the accident domain
    create_trapezoidal_number(pAccidentDomain, "low", -0.1, 0.0, 0.1, 0.2);
    create_trapezoidal_number(pAccidentDomain, "medium", 0.1, 0.2, 0.7, 0.8);

    create_trapezoidal_number(pAccidentDomain, "high", 0.7, 0.8, 0.9, 1.0);

    // Creating rules for fuzzy logic
    PyObject *pBaseRule = PyObject_GetAttrString(pFuzzyModule, "BaseRule");
    PyObject *rules = PyList_New(0);

    // Adding rules
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

    // Creating an instance of FuzzyInference, using Py_BuildValue,
    // to create a dict in Python (hash table data structure)

    // Creating an instance of FuzzyInference
    PyObject *pFuzzyInferenceClass = PyObject_GetAttrString(pFuzzyModule, "FuzzyInference");
    PyObject *fuzzyInference = PyObject_CallObject(
        pFuzzyInferenceClass,
        PyTuple_Pack(2,
                     Py_BuildValue("{s:O, s:O}", "age", pAgeDomain, "accident", pAccidentDomain),
                     rules));

    // Performing the calculation, the PyObject object is returned *,
    // in the Python FuzzyNumbers class, you can defuzzify the value,
    // since this fuzzy inference algorithm assumes
    // a defuzzified value as the result
    PyObject *pResult
        = PyObject_CallMethod(fuzzyInference, "compute", "O", Py_BuildValue("{s:i}", "age", age));

    // Freeing up resources
    // Py_DECREF(pResult);
    Py_DECREF(fuzzyInference);
    Py_DECREF(pAgeDomain);
    Py_DECREF(pAccidentDomain);
    Py_DECREF(pBaseRule);
    Py_DECREF(pFuzzyModule);

    return pResult;
}

// Function for obtaining the defuzzified value of an fuzzy number
double get_defuzz_value(PyObject *f_num) {
    double result = 0.0;
    // Getting the method __float__
    PyObject *pFloatMethod = PyObject_GetAttrString(f_num, "defuzz");
    if (pFloatMethod) {
        // Calling the method __float__()
        PyObject *pFloatValue = PyObject_CallObject(pFloatMethod, NULL);
        if (pFloatValue) {
            std::cout << PyFloat_AsDouble(pFloatValue) << std::endl;
            // Converting the result to double
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