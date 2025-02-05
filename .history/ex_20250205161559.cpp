#include <iostream>
#include <Python.h>


int main() {
    // Инициализация интерпретатора Python
    Py_Initialize();

    // Импортируем модуль fan
    PyObject *pName = PyUnicode_FromString("fuzzyops.fan.fan");
    PyObject *pModule = PyImport_Import(pName);
    Py_DECREF(pName);

}