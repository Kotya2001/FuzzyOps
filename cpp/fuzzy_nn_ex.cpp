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

long get_prediction(PyObject *pModel, std::vector<double> input, const char *device) {
    // Creating an input tensor as a 2D tensor
    // Creating a list that will be a 2D array
    PyObject *pInputList = PyList_New(1); // A list of one element
    PyObject *pInnerList = PyList_New(2); // An internal list with two values
    PyList_SetItem(pInnerList, 0, PyFloat_FromDouble(input[0]));
    PyList_SetItem(pInnerList, 1, PyFloat_FromDouble(input[1]));
    PyList_SetItem(
        pInputList, 0, pInnerList); // Setting the internal list as the first element

    // Creating a tensor from a 2D list
    PyObject *pTensor = PyObject_CallMethod(import_module("torch"), "tensor", "O", pInputList);
    PyObject *pDevice
        = PyObject_CallMethod(import_module("torch"), "device", "O", PyUnicode_FromString(device));

    // The tensor also needs to be transferred to the device on which the model is based.
    PyObject *pToDeviceMethod = PyObject_GetAttrString(pTensor, "to");

    PyObject *pTensorToDevice = PyObject_CallObject(pToDeviceMethod, PyTuple_Pack(1, pDevice));

    // Calling the __call__ method for pModel
    PyObject *pCallMethod = PyObject_GetAttrString(pModel, "__call__");
    if (!pCallMethod || !PyCallable_Check(pCallMethod)) {
        std::cerr << "'__call__' method not found or is not callable." << std::endl;
        Py_DECREF(pInputList);
        return -1;
    }

    // Calling a tensor model
    PyObject *pRes = PyObject_CallObject(pCallMethod, PyTuple_Pack(1, pTensorToDevice));
    if (!pRes) {
        PyErr_Print();
        std::cerr << "Failed to call the model." << std::endl;
        Py_DECREF(pInputList);
        Py_DECREF(pTensor);
        return -1;
    }

    // Getting the index of the maximum value
    PyObject *pArgmax = PyObject_CallMethod(pRes, "argmax", "i", 1); // We use dim=1
    if (!pArgmax) {
        PyErr_Print();
        std::cerr << "Failed to call 'argmax' on the result." << std::endl;
        Py_DECREF(pRes);
        return -1;
    }

    // Converting the result to an integer
    long predicted_class = PyLong_AsLong(pArgmax);
    if (predicted_class == -1 && PyErr_Occurred()) {
        PyErr_Print();
        std::cerr << "Failed to convert prediction to long." << std::endl;
        Py_DECREF(pArgmax);
        Py_DECREF(pRes);
        return -1;
    }

    // Freeing up resources
    Py_DECREF(pInputList);
    Py_DECREF(pTensor);
    Py_DECREF(pRes);
    Py_DECREF(pArgmax);

    return predicted_class;
}

PyObject *train_model(const std::string &path, const char *device) {
    // Import the fuzzyops module and run the script
    PyObject *pModule = import_module("fuzzyops.fuzzy_nn");
    if (!pModule)
        return nullptr;

    // Preparing arguments for the process_csv_data function
    int n_features = 2;
    PyObject *pArgs = PyTuple_Pack(5,
                                   PyUnicode_FromString(path.c_str()),
                                   PyUnicode_FromString("Species"),
                                   PyLong_FromLong(n_features),
                                   Py_True,  // use_label_encoder
                                   Py_True); // drop_index

    // Calling the process_csv_data function
    // Get a reference to the process_csv_data function
    PyObject *pProcessFunc = PyObject_GetAttrString(pModule, "process_csv_data");
    if (!pProcessFunc || !PyCallable_Check(pProcessFunc)) {
        if (pProcessFunc) {
            std::cerr << "Attribute 'process_csv_data' is not callable." << std::endl;
        } else {

            std::cerr << "Attribute 'process_csv_data' does not exist." << std::endl;
        }
        Py_DECREF(pArgs);
        Py_DECREF(pModule);
        return nullptr;
    }
    PyObject *pData = PyObject_CallObject(pProcessFunc, pArgs);

    // Freeing up memory
    Py_DECREF(pArgs);
    Py_DECREF(pProcessFunc);

    if (!pData) {
        PyErr_Print();
        std::cerr << "Failed to call process_csv_data" << std::endl;
        Py_DECREF(pModule);
        return nullptr;
    }

    // We get X and y
    PyObject *X = PyTuple_GetItem(pData, 0);
    PyObject *y = PyTuple_GetItem(pData, 1);

    PyObject *n_terms = PyList_New(2);
    PyList_SetItem(n_terms, 0, PyLong_FromLong(5)); // The first value
    PyList_SetItem(n_terms, 1, PyLong_FromLong(5)); // The second value

    // Packing arguments to create a model instance
    PyObject *pModelArgs = PyTuple_Pack(10,
                                        X,
                                        y,
                                        n_terms,
                                        PyLong_FromLong(3),       // Now we are passing the list
                                        PyFloat_FromDouble(3e-4), // lr
                                        PyUnicode_FromString("classification"), // task_type
                                        PyLong_FromLong(8),                     // batch_size
                                        PyUnicode_FromString("gauss"),          // member_func_type
                                        PyLong_FromLong(100),                   // epochs
                                        Py_True                                 // verbose
    );

    // Creating a model
    PyObject *pModelClass = PyObject_GetAttrString(pModule, "Model");
    PyObject *pModel = PyObject_CallObject(pModelClass, pModelArgs);
    if (!pModel) {
        PyErr_Print();
        std::cerr << "Failed to create model instance." << std::endl;
        Py_DECREF(pModelArgs);
        return nullptr;
    }

    PyObject_SetAttrString(pModel, "device", PyUnicode_FromString(device));

    // We get a link to the train method
    PyObject *pTrainMethod = PyObject_GetAttrString(pModel, "train");
    if (!pTrainMethod || !PyCallable_Check(pTrainMethod)) {
        if (pTrainMethod) {
            std::cerr << "'train' is not callable." << std::endl;
        } else {
            std::cerr << "'train' method not found." << std::endl;
        }
        Py_DECREF(pModel); // We release the pModel, if necessary.
        return nullptr;
    }

    // Calling the train method without arguments
    PyObject *m = PyObject_CallObject(pTrainMethod, NULL);
    if (!m) {
        PyErr_Print();
        std::cerr << "Failed to call 'train' method." << std::endl;
        Py_DECREF(pTrainMethod);
        Py_DECREF(pModel); // We release the pModel, if necessary.
        return nullptr;
    }

    Py_DECREF(pData);
    Py_DECREF(pModel);
    Py_DECREF(pModelArgs);
    Py_DECREF(pTrainMethod);
    Py_DECREF(pModule);

    return m;
}

int main(int argc, char *argv[]) {
    initialize_python();
    const char *device = "cpu";
    // Path to the Iris.csv data in the cpp directory
    std::string path = "Iris.csv";
    // Input data for the model after training
    std::vector<double> inputs = {5.1, 3.5};
    // Model training
    PyObject *model = train_model(path, device);
    // Getting a prediction from a model
    long cls = get_prediction(model, inputs, device);

    std::cout << "Predicted class: " << cls << std::endl;

    Py_DECREF(model);
    Py_Finalize();
    return 0;
}