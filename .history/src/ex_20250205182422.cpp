#include <Python.h>
#include <iostream>

int main() {
    // Инициализация интерпретатора Python
    Py_Initialize();

    // Импортируем модуль fan
    PyObject *pName = PyUnicode_FromString("fuzzyops.fan.fan");
    PyObject *pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    Py_Finalize();
    return 0;
}