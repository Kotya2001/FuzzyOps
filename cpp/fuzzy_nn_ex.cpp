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

long get_prediction(PyObject *pModel, std::vector<double> input, const char *device) {
    // Создание входного тензора как 2D тензор
    // Создаем список, который будет представлять собой 2D массив
    PyObject *pInputList = PyList_New(1); // Список из одного элемента
    PyObject *pInnerList = PyList_New(2); // Внутренний список с двумя значениями
    PyList_SetItem(pInnerList, 0, PyFloat_FromDouble(input[0]));
    PyList_SetItem(pInnerList, 1, PyFloat_FromDouble(input[1]));
    PyList_SetItem(
        pInputList, 0, pInnerList); // Устанавливаем внутренний список в качестве первого элемента

    // Создаем тензор из 2D списка
    PyObject *pTensor = PyObject_CallMethod(import_module("torch"), "tensor", "O", pInputList);
    PyObject *pDevice
        = PyObject_CallMethod(import_module("torch"), "device", "O", PyUnicode_FromString(device));

    // Тензор также необходимо перенести на device, на котором лежит модель
    PyObject *pToDeviceMethod = PyObject_GetAttrString(pTensor, "to");

    PyObject *pTensorToDevice = PyObject_CallObject(pToDeviceMethod, PyTuple_Pack(1, pDevice));

    // Вызов метода __call__ у pModel
    PyObject *pCallMethod = PyObject_GetAttrString(pModel, "__call__");
    if (!pCallMethod || !PyCallable_Check(pCallMethod)) {
        std::cerr << "'__call__' method not found or is not callable." << std::endl;
        Py_DECREF(pInputList);
        return -1;
    }

    // Вызов модели с тензором
    PyObject *pRes = PyObject_CallObject(pCallMethod, PyTuple_Pack(1, pTensorToDevice));
    if (!pRes) {
        PyErr_Print();
        std::cerr << "Failed to call the model." << std::endl;
        Py_DECREF(pInputList);
        Py_DECREF(pTensor);
        return -1;
    }

    // Получение индекса максимального значения
    PyObject *pArgmax = PyObject_CallMethod(pRes, "argmax", "i", 1); // Используем dim=1
    if (!pArgmax) {
        PyErr_Print();
        std::cerr << "Failed to call 'argmax' on the result." << std::endl;
        Py_DECREF(pRes);
        return -1;
    }

    // Преобразование результата в целое число
    long predicted_class = PyLong_AsLong(pArgmax);
    if (predicted_class == -1 && PyErr_Occurred()) {
        PyErr_Print();
        std::cerr << "Failed to convert prediction to long." << std::endl;
        Py_DECREF(pArgmax);
        Py_DECREF(pRes);
        return -1;
    }

    // Освобождение ресурсов
    Py_DECREF(pInputList);
    Py_DECREF(pTensor);
    Py_DECREF(pRes);
    Py_DECREF(pArgmax);

    return predicted_class;
}

PyObject *train_model(const std::string &path, const char *device) {
    // Импортируем модуль fuzzyops и запускаем скрипт
    PyObject *pModule = import_module("fuzzyops.fuzzy_nn");
    if (!pModule)
        return nullptr;

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
        return nullptr;
    }
    PyObject *pData = PyObject_CallObject(pProcessFunc, pArgs);

    // Освобождаем память
    Py_DECREF(pArgs);
    Py_DECREF(pProcessFunc);

    if (!pData) {
        PyErr_Print();
        std::cerr << "Failed to call process_csv_data" << std::endl;
        Py_DECREF(pModule);
        return nullptr;
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
        return nullptr;
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
        return nullptr;
    }

    // Вызываем метод train без аргументов
    PyObject *m = PyObject_CallObject(pTrainMethod, NULL);
    if (!m) {
        PyErr_Print();
        std::cerr << "Failed to call 'train' method." << std::endl;
        Py_DECREF(pTrainMethod);
        Py_DECREF(pModel); // Освобождаем pModel, если это необходимо
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
    // путь к данным Iris.csv в директории cpp
    std::string path = "Iris.csv";
    // входные данные для модели после обучения
    std::vector<double> inputs = {5.1, 3.5};
    // обучение модели
    PyObject *model = train_model(path, device);
    // получение предсказания у модели
    long cls = get_prediction(model, inputs, device);

    std::cout << "Predicted class: " << cls << std::endl;

    Py_DECREF(model);
    Py_Finalize();
    return 0;
}