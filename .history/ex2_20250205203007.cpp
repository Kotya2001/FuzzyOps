#include <Python.h>
#include <iostream>

PyObject *import_module(const char *module_name) {
    PyObject *pName = PyUnicode_FromString(module_name);
    PyObject *pModule = PyImport_Import(pName);
    Py_DECREF(pName);
    return pModule;
}
