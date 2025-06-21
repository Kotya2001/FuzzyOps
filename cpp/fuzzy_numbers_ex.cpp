#include <Python.h>
#include <cstdio>
#include <iostream>
#include <vector>

std::string format_string(const char *format, int value) {
    char buffer[50]; // A large enough buffer to store the string
    snprintf(buffer, sizeof(buffer), format, value);
    return std::string(buffer);
}

void initialize_python() { Py_Initialize(); }

// Function for importing a module from the fuzzyops library
PyObject *import_module(const char *module_name) {
    PyObject *pName = PyUnicode_FromString(module_name);
    PyObject *pModule = PyImport_Import(pName);
    Py_DECREF(pName);
    if (!pModule) {
        PyErr_Print();
        std::cerr << "Failed to load module" << std::endl;
        Py_DECREF(pModule);
        return {};
    }
    return pModule;
}

// Function for creating a domain
PyObject *create_domain(double min_val, double max_val, const char *name) {
    // Импортируем модуль fuzzyops.fuzzy_numbers
    PyObject *pDomainModule = import_module("fuzzyops.fuzzy_numbers");
    PyObject *pDomainClass = PyObject_GetAttrString(pDomainModule, "Domain");

    // Creating a tuple with parameters
    PyObject *pArgs = PyTuple_Pack(2, PyFloat_FromDouble(min_val), PyFloat_FromDouble(max_val));

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

// A function for creating a Gaussian number
void create_gauss_number(PyObject *pDomain, const char *name, double sigma, double mean) {
    PyObject *pCreateNumber = PyObject_GetAttrString(pDomain, "create_number");
    PyObject *pArgs = PyTuple_Pack(
        3, PyUnicode_FromString("gauss"), PyFloat_FromDouble(sigma), PyFloat_FromDouble(mean));

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

// A function for summing fuzzy numbers and getting the final fuzzy number
PyObject *sum_numbers(PyObject *pDomain) {

    PyObject *summator = PyObject_GetAttrString(pDomain, "out");
    if (!summator) {
        PyErr_Print();
        std::cerr << "Failed to get summator" << std::endl;
    }

    // We start with a zero value for the final result.
    PyObject *total_sum = summator; // creating a copy of summator
    if (!total_sum) {
        PyErr_Print();
        std::cerr << "Failed to initialize total_sum" << std::endl;
        Py_DECREF(summator);
        return NULL; // We return NULL in case of an error
    }

    for (int i = 0; i < 50; i++) {

        std::string nNameStr = format_string("n%d", i);
        create_gauss_number(pDomain, nNameStr.c_str(), 1.0, i);

        PyObject *pAttrib1 = PyObject_GetAttrString(pDomain, nNameStr.c_str());
        if (!pAttrib1) {
            PyErr_Print();
            std::cerr << "Failed to get value on domain: " << nNameStr << std::endl;
        }
        PyObject *pSumMethod = PyObject_GetAttrString(summator, "__iadd__");
        if (pSumMethod) {
            PyObject *pResult = PyObject_CallFunctionObjArgs(pSumMethod, pAttrib1, NULL);
            if (!pResult) {
                PyErr_Print();
                std::cerr << "Failed to add value to out" << std::endl;
            } else {
                // Updating the total amount
                PyObject *tmp_sum = total_sum; // Saving the current amount
                total_sum = pResult;           // Updating the amount

                // Releasing the old amount
                Py_DECREF(tmp_sum);
            }
            Py_DECREF(pSumMethod); // Releasing the addition method
        } else {
            PyErr_Print();
            std::cerr << "Failed to get summation method" << std::endl;
        }

        Py_DECREF(pAttrib1);
    }
    return total_sum;
}
// A function for obtaining the degrees of confidence of the result
std::vector<double> print_values(PyObject *fNum, const char *device) {
    std::vector<double> result;

    PyObject *pDomain = PyObject_GetAttrString(fNum, "domain");
    if (!pDomain) {
        PyErr_Print();
        return {}; // Returning an empty vector in case of an error
    }

    // Calling the method .to() with the 'cpu' argument
    PyObject *pToMethod = PyObject_GetAttrString(pDomain, "to");
    if (pToMethod) {
        PyObject *pDevice = PyUnicode_FromString(device);
        PyObject *pOutDevice = PyObject_CallFunctionObjArgs(pToMethod, pDevice, NULL);
        Py_DECREF(pDevice);
        Py_DECREF(pToMethod);

        if (!pOutDevice) {
            PyErr_Print();
            return {};
        }

        // We get the values from the tensor
        PyObject *pValues = PyObject_GetAttrString(fNum, "values");
        if (pValues) {
            // Checking whether an object is a tensor
            if (PyObject_HasAttrString(pValues, "tolist")) {
                // Converting a tensor to a regular Python list
                PyObject *pListMethod = PyObject_GetAttrString(pValues, "tolist");
                if (pListMethod) {
                    PyObject *pList = PyObject_CallObject(pListMethod, NULL);
                    if (pList) {
                        // Getting the size of the list
                        Py_ssize_t size = PyList_Size(pList);
                        // Saving the values to a C++ vector
                        result.reserve(size);
                        for (Py_ssize_t i = 0; i < size; ++i) {
                            PyObject *pItem = PyList_GetItem(pList, i);
                            if (pItem) {
                                double value = PyFloat_AsDouble(pItem);
                                result.push_back(value);
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

            Py_DECREF(pValues);
        } else {
            PyErr_Print();
        }

        Py_DECREF(pOutDevice);
    } else {
        PyErr_Print();
    }
    return result;
}
// A function for obtaining a defuzzified value of a fuzzy number
double print_f_num(PyObject *f_num) {
    double result = 0.0;

    // Getting the method __float__
    PyObject *pFloatMethod = PyObject_GetAttrString(f_num, "__float__");
    if (pFloatMethod) {
        // Calling the method __float__()
        PyObject *pFloatValue = PyObject_CallObject(pFloatMethod, NULL);
        if (pFloatValue) {
            // Converting the result to a double
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

    PyObject *pStrMethod = PyObject_GetAttrString(f_num, "__str__");
    if (pStrMethod) {
        // Calling the method __str__()
        PyObject *pStrRep = PyObject_CallObject(pStrMethod, NULL);
        if (pStrRep) {
            // We output the string representation to the console
            std::cout << "String representation of f_num " << PyUnicode_AsUTF8(pStrRep)
                      << std::endl;
            // Freeing up resources
            Py_DECREF(pStrRep);
        } else {
            PyErr_Print();
            std::cerr << "Failed to call __str__" << std::endl;
        }
        Py_DECREF(pStrMethod);
    } else {
        PyErr_Print();
        std::cerr << "Failed to get __str__ method" << std::endl;
    }
    return result;
};

int main() {
    initialize_python();
    // Creating a Domain
    PyObject *pDomain = create_domain(0, 101, "d");
    // Creating a fuzzy number
    create_gauss_number(pDomain, "out", 1.0, 0.0);
    // Получение результата суммирования
    PyObject *result = sum_numbers(pDomain);

    // Getting a representation of a fuzzy number from Python
    // (A defuzzified value is returned in Python)
    double f_num_value = print_f_num(result);
    std::cout << "Value of f_num: " << f_num_value << std::endl;

    // Obtaining degrees of confidence in the result of the sum
    std::vector<double> values = print_values(result, "cpu");
    std::cout << "Values: ";
    for (double value : values) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    if (pDomain) {
        Py_DECREF(pDomain);
    }
    Py_DECREF(result);

    Py_Finalize();
}