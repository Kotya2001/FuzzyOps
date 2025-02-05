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
