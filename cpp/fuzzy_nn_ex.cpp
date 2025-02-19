#include <Python.h>
#include <iostream>
#include <string>
#include <vector>

// Инициализация Python
void initialize_python() { Py_Initialize(); }

// Импорт модуля Python
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

void get_prediction(PyObject *pModel) {
    // Создание входного тензора как 2D тензор
    // Создаем список, который будет представлять собой 2D массив
    PyObject *pInputList = PyList_New(1); // Список из одного элемента
    PyObject *pInnerList = PyList_New(2); // Внутренний список с двумя значениями
    PyList_SetItem(pInnerList, 0, PyFloat_FromDouble(5.1));
    PyList_SetItem(pInnerList, 1, PyFloat_FromDouble(3.5));
    PyList_SetItem(
        pInputList, 0, pInnerList); // Устанавливаем внутренний список в качестве первого элемента

    // Создаем тензор из 2D списка
    PyObject *pTensor = PyObject_CallMethod(import_module("torch"), "tensor", "O", pInputList);

    // Вызов метода __call__ у pModel
    PyObject *pCallMethod = PyObject_GetAttrString(pModel, "__call__");
    if (!pCallMethod || !PyCallable_Check(pCallMethod)) {
        std::cerr << "'__call__' method not found or is not callable." << std::endl;
        Py_DECREF(pInputList);
        return;
    }

    // Вызов модели с тензором
    PyObject *pRes = PyObject_CallObject(pCallMethod, PyTuple_Pack(1, pTensor));
    if (!pRes) {
        PyErr_Print();
        std::cerr << "Failed to call the model." << std::endl;
        Py_DECREF(pInputList);
        Py_DECREF(pTensor);
        return;
    }

    // Получение индекса максимального значения
    PyObject *pArgmax = PyObject_CallMethod(pRes, "argmax", "i", 1); // Используем dim=1
    if (!pArgmax) {
        PyErr_Print();
        std::cerr << "Failed to call 'argmax' on the result." << std::endl;
        Py_DECREF(pRes);
        return;
    }

    // Преобразование результата в целое число
    long predicted_class = PyLong_AsLong(pArgmax);
    if (predicted_class == -1 && PyErr_Occurred()) {
        PyErr_Print();
        std::cerr << "Failed to convert prediction to long." << std::endl;
        Py_DECREF(pArgmax);
        Py_DECREF(pRes);
        return;
    }

    std::cout << "Predicted class: " << predicted_class << std::endl;

    // Освобождение ресурсов
    Py_DECREF(pInputList);
    Py_DECREF(pTensor);
    Py_DECREF(pRes);
    Py_DECREF(pArgmax);
}

void run_fuzzyops(const std::string &path, const char *device) {
    // Импортируем модуль fuzzyops и запускаем скрипт
    PyObject *pModule = import_module("fuzzyops.fuzzy_nn");
    if (!pModule)
        return;

    // Проверяем наличие атрибута (функции) process_csv_data
    if (!PyObject_HasAttrString(pModule, "process_csv_data")) {
        std::cerr << "Function 'process_csv_data' does not exist in the module." << std::endl;
        Py_DECREF(pModule);
        return;
    }

    // Подготовка аргументов для функции process_csv_data
    int n_features = 2;
    PyObject *pArgs = PyTuple_Pack(5,
                                   PyUnicode_FromString(path.c_str()),
                                   PyUnicode_FromString("Species"),
                                   PyLong_FromLong(n_features),
                                   Py_True,  // use_label_encoder
                                   Py_True); // drop_index

    // Вызываем функцию process_csv_data
    // Получаем ссылку на функцию process_csv_data
    PyObject *pProcessFunc = PyObject_GetAttrString(pModule, "process_csv_data");
    if (!pProcessFunc || !PyCallable_Check(pProcessFunc)) {
        if (pProcessFunc) {
            std::cerr << "Attribute 'process_csv_data' is not callable." << std::endl;
        } else {

            std::cerr << "Attribute 'process_csv_data' does not exist." << std::endl;
        }
        Py_DECREF(pArgs);
        Py_DECREF(pModule);
        return;
    }
    PyObject *pData = PyObject_CallObject(pProcessFunc, pArgs);

    // Освобождаем память
    Py_DECREF(pArgs);
    Py_DECREF(pProcessFunc);

    if (!pData) {
        PyErr_Print();
        std::cerr << "Failed to call process_csv_data" << std::endl;
        Py_DECREF(pModule);
        return;
    }

    // Получаем X и y
    PyObject *X = PyTuple_GetItem(pData, 0);
    PyObject *y = PyTuple_GetItem(pData, 1);

    PyObject *n_terms = PyList_New(2);
    PyList_SetItem(n_terms, 0, PyLong_FromLong(5)); // Первое значение
    PyList_SetItem(n_terms, 1, PyLong_FromLong(5)); // Второе значение

    // Пакуем аргументы для создания экземпляра модели
    PyObject *pModelArgs = PyTuple_Pack(10,
                                        X,
                                        y,
                                        n_terms,
                                        PyLong_FromLong(3),       // Теперь передаем список
                                        PyFloat_FromDouble(3e-4), // lr
                                        PyUnicode_FromString("classification"), // task_type
                                        PyLong_FromLong(8),                     // batch_size
                                        PyUnicode_FromString("gauss"),          // member_func_type
                                        PyLong_FromLong(100),                   // epochs
                                        Py_True                                 // verbose
    );

    // Создаем модель
    PyObject *pModelClass = PyObject_GetAttrString(pModule, "Model");
    PyObject *pModel = PyObject_CallObject(pModelClass, pModelArgs);
    if (!pModel) {
        PyErr_Print();
        std::cerr << "Failed to create model instance." << std::endl;
        Py_DECREF(pModelArgs);
        return;
    }

    PyObject_SetAttrString(pModel, "device", PyUnicode_FromString(device));

    // Получаем ссылку на метод train
    PyObject *pTrainMethod = PyObject_GetAttrString(pModel, "train");
    if (!pTrainMethod || !PyCallable_Check(pTrainMethod)) {
        if (pTrainMethod) {
            std::cerr << "'train' is not callable." << std::endl;
        } else {
            std::cerr << "'train' method not found." << std::endl;
        }
        Py_DECREF(pModel); // Освобождаем pModel, если это необходимо
        return;
    }

    // Вызываем метод train без аргументов
    PyObject *m = PyObject_CallObject(pTrainMethod, NULL);
    if (!m) {
        PyErr_Print();
        std::cerr << "Failed to call 'train' method." << std::endl;
        Py_DECREF(pTrainMethod);
        Py_DECREF(pModel); // Освобождаем pModel, если это необходимо
        return;
    }

    // Получение предсказания
    get_prediction(m);

    // Освобождаем память
    Py_DECREF(pData);
    Py_DECREF(m);
    Py_DECREF(pModel);
    Py_DECREF(pModelArgs);
    Py_DECREF(pTrainMethod);
    Py_DECREF(pModule);
}

int main(int argc, char *argv[]) {
    initialize_python();
    std::string path = "/Users/ilabelozerov/FuzzyOps/src/fuzzyops/tests/Iris.csv";
    run_fuzzyops(path, "cpu");
    Py_Finalize();
    return 0;
}