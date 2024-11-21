# FuzzyOps
Библиотека алгоритмов нечеткого прогнозирования и поддержки принятия решений

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

### Запуск тестов

После устоновки запуск тестов осуществляется:

 - ```python -m unittest fuzzyops.tests.test_fan```
 - test_fan название модуля,
   далее можно добавить название класса, и название конкретной функции
   Например,
   ```python -m unittest fuzzyops.tests.test_fan.TestFAN.testSimpleGraph```

