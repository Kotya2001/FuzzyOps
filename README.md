# FuzzyOps
Библиотека алгоритмов нечеткого прогнозирования и поддержки принятия решений

Библиотека предназначена для применения:
- в научных лабораториях, занимающихся исследованиями в области многокритериального анализа, оптимального планирования и управления;
- в конструкторских бюро, занимающихся проектированием сложных технических систем;
- в компаниях занимающихся разработкой систем поддержки принятия решений. Фактически библиотека должна использоваться при создании как полнофункциональных программных продуктов, так и экспериментальных макетов программных комплексов, предназначенных для работы с нечеткими факторами.

Библиотекой можно также пользоваться путем прямого вызова функций в программах на С++, следуя инструкции:
- https://github.com/Kotya2001/FuzzyOps/blob/main/cpp/README.md

Также возможно реализовывать в своем ПО обращение к библиотеки по RESTfull API (развертывание веб-сервиса происходит на своих ресурсах), следуя следующим инструкциями:
 * https://github.com/Kotya2001/FuzzyOps-App - исходный код веб-сервиса для развертывания;
 * https://github.com/Kotya2001/FuzzyOps-App/tree/main/posters - примеры для реализации обращения к веб-сервису по API (примеры реализованы на Python);
 * https://github.com/Kotya2001/FuzzyOps-App/wiki/Инструкция-по-использованию-алгоритмов-в-веб%E2%80%90серсиве-(по-API) - инструкция по использованию алгоритмов библиотеки по RESTful API.


### Как установить библиотеку

Для установки библиотеки в качестве пип-пакета, необходимо использовать
команду: `pip install git+https://{login}:{token}@github.com/Kotya2001/FuzzyOps.git`,
подставив соответствующие значения:

  - login: ваш логин на GitHub 
  - token: как создать токен - [тут](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)

Или
 - ```pip install fuzzyops```

### Перед установкой

Создайте виртуальное окружение с Python >= 3.10

  ```Полный путь к исполняемому файлу Python 3.10 -m venv env```

Активация окружения

  - Macos: ```source env/bin/activate```
  - Windows: ```.\env\Scripts\activate```
  - Linux: ```source env/bin/activate```

Установка Cuda Toolkit 11.5

  - https://developer.nvidia.com/cuda-11-5-0-download-archive

Установите PyTorch в зависимости от вашей операционной сисиемы

  - Windows: ```pip3 install torch --index-url https://download.pytorch.org/whl/cu117```
  - Macos: ```pip3 install torch```
  - Linux: ```pip3 install torch```

### Минимальные технические требования

- Объем ОЗУ не менее 2 гб;
- Для вычислений на CUDA устройство графического вывода GeForce RTX 3070 и выше;
- Установленные Python версии 3.10 или выше

### Инструкция по работе с библиотекой и документация к исходному коду библиотеки:

-  Инструкция по работе с библиотекой - https://github.com/Kotya2001/FuzzyOps/wiki/Инструкция-по-работе-с-библиотекой-FuzzyOps;
-  Документация к исходному коду библиотеки - https://fuzzyops.readthedocs.io/en/latest/

### Запуск тестов

После устоновки запуск тестов осуществляется согласно инструкции:

 - Инструкция по запуску тестов - https://github.com/Kotya2001/FuzzyOps/wiki/Инструкция-по-запуску-тестов-библиотеки-FuzzyOps
   

### Инструкция по использованию библиотеки в С++ программах

-  Инструкция по использованию библиотека в C++ программах - https://github.com/Kotya2001/FuzzyOps/blob/main/cpp/README.md


### Веб-сервис для обращения к алгоритмам библиотеки по RESTfull API:

- Искодный код веб-севиса и инструкции к его использованию - https://github.com/Kotya2001/FuzzyOps-App


### Описание папок с файлами репозитория библиотеки

 * [cpp](https://github.com/Kotya2001/FuzzyOps/tree/main/cpp) - Инструкция по использованию библиотеки в С++ программах и примеры кода использования библиотеки на python и на С++;
 * [dist](https://github.com/Kotya2001/FuzzyOps/tree/main/dist) - Установочные файлы библиотеки (дистрибутивы);
 * [docs](https://github.com/Kotya2001/FuzzyOps/tree/main/docs) - Файлы, формата .html с документацией к исходному коду (собранные с помощью библиотеки [sphinx](https://www.sphinx-doc.org/en/master/))
 * [example](https://github.com/Kotya2001/FuzzyOps/tree/main/examples):
   * [common](https://github.com/Kotya2001/FuzzyOps/tree/main/examples/common) - Примеры использования кома библиотеки;
   * Остальные файлы - практические примеры использования кода библиотеки;
 * [src](https://github.com/Kotya2001/FuzzyOps/tree/main/src) - Исходные коды библиотеки:
   * [docs](https://github.com/Kotya2001/FuzzyOps/tree/main/src/docs) - Файлы, формата .html с документацией к исходному коду (собранные с помощью библиотеки [sphinx](https://www.sphinx-doc.org/en/master/));
   * [fuzzyops](https://github.com/Kotya2001/FuzzyOps/tree/main/src/fuzzyops) - Исходные коды библиотеки:
     * [fan](https://github.com/Kotya2001/FuzzyOps/tree/main/src/fuzzyops/fan) - Исходные коды нечетких аналитических сетей;
     * [fuzzy_logic](https://github.com/Kotya2001/FuzzyOps/tree/main/src/fuzzyops/fuzzy_logic) Исходные коды алгоритмов нечеткой логики;
     * [fuzzy_msa](https://github.com/Kotya2001/FuzzyOps/tree/main/src/fuzzyops/fuzzy_msa) - Исходные коды классических алгоритмов многокритериального анализа с нечеткими переменными;
     * [fuzzy_neural_net](https://github.com/Kotya2001/FuzzyOps/tree/main/src/fuzzyops/fuzzy_neural_net) - Исходные коды алгоритмов нечетких нейронных сетей (второй варинат алгоритмов);
     * [fuzzy_nn](https://github.com/Kotya2001/FuzzyOps/tree/main/src/fuzzyops/fuzzy_nn) - Исходные коды алгоритмов нечетких нейронных сетей (Сеть ANFIS);
     * [fuzzy_numbers](https://github.com/Kotya2001/FuzzyOps/tree/main/src/fuzzyops/fuzzy_numbers/fuzzify) - Исходные коды реализации нечетких чисел (фаззификация, дефаззификация, нечеткая арифметика);
     * [fuzzygraphs](https://github.com/Kotya2001/FuzzyOps/tree/main/src/fuzzyops/graphs/fuzzgraph) - Исходные коды реализации нечетких графов;
     * [fuzzygraphs_algs](https://github.com/Kotya2001/FuzzyOps/tree/main/src/fuzzyops/graphs/algorithms) - Исходные коды алгоритмов на нечетких графах (Отношения нечеткого доминирования, нечеткие факторные модели, нечеткие транспортные графы);
     * [fuzzy_pred](https://github.com/Kotya2001/FuzzyOps/tree/main/src/fuzzyops/prediction) - Исходные коды алгоритмов нечеткого прогнозирования;
     * [sequencing_assignment](https://github.com/Kotya2001/FuzzyOps/tree/main/src/fuzzyops/sequencing_assignment) - Исходные коды алгоритмов на нечетких графах последовательности работ в задачах о назначении;
     * [tests](https://github.com/Kotya2001/FuzzyOps/tree/main/src/fuzzyops/tests) - Коды тестов алгоритмов.
 * [readthedocs](https://github.com/Kotya2001/FuzzyOps/blob/main/.readthedocs.yml) - Файл для автоматической сборки и размещений документации на https://about.readthedocs.com;
 * [doc_reqs.txt](https://github.com/Kotya2001/FuzzyOps/blob/main/doc_reqs.txt) - Файл с зависимостими библиотеки для сборки документации на https://about.readthedocs.com;
 * [requirements](https://github.com/Kotya2001/FuzzyOps/blob/main/requirements.txt) - Файл зависимостями для установки библиотеки;
 * [setup.cfg](https://github.com/Kotya2001/FuzzyOps/blob/main/setup.cfg) - Конфигурационный файл для сборки дистрибутива библиотеки;
 * [setup.py](https://github.com/Kotya2001/FuzzyOps/blob/main/setup.py) - Файл для сборки дистрибутива библиотеки с помощью `setuptools`;
 * [LICENSE](https://github.com/Kotya2001/FuzzyOps/blob/main/LICENSE) - Файл лицензии библиотеки;
   