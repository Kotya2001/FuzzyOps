"""
Задача:
При разработке каких-либо устройств
необходимо сравнить разработку с конкурентами и оценить стоимость изделия,
имея набор признаков уже существующий техники.

Так, например, используя данные с сайта https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification
о характеристиках мобильных
телефонах можно предсказать по входным данных характеристик ценовой диапазон (категории от 0 до 3), то есть категорию
к который отнесется новое устройство (например, при разработке нового мобильного телефона)

Ценовые категории должны быть заданы до обучения алгоритма, чтобы при использовании получить метку класса и понимать
какая конкретная стоимость может быть у нового изделия.

Таким образом, решается задачи классификаии с учителем при помощи алгоритма нечеткой нейронной сети (ANFIS)

Для тренировки используются матрица объекты-признаки, состоящая из 15 первых признаков:

    Мощность батареи;
    Наличие/отсутствие технологии Bluetooth (бинарные признак);
    Скорость процессора в миллисекундах;
    Поддержка двойной сим-карты (бинарные признак);
    Количество мега-пикселей у фронтальной камеры;
    Поддержка технологии 4G (бинарные признак);
    Объем памяти (Гб) (не RAM);
    Толщина экрана (см);
    Вес телефона (г);
    Число ядер процессора;
    Количество мега-пикселей у задней камеры;
    Разрешение экрана в пикселях (высота);
    Разрешение экрана в пикселях (ширина);
    Объем оперативной памяти;
    Высота экрана (см);
    Ширина экрана (см);

После обучения моделе необходимо подать вектор из значений каждого признака (число перечисленно выше)
на воходе получится метка класса той ценовой категории, к которой принадлежит устройство.

"""

# (Библиотека уже установлена в ваш проект)
from fuzzyops.fuzzy_nn import Model
from sklearn.model_selection import train_test_split

import pandas as pd
import torch

# Загружаем необходимые данные и немного предобрабатываем их
df = pd.read_csv("train.csv")
Y = df["price_range"]
X = df.drop("price_range", axis=1)

# Разделим выборки на обучение и тест
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
n_features = 15

# Преобразуем данные к torch.Tensor
x = torch.Tensor(X_test.iloc[:, 0: n_features].values)
y = torch.Tensor(Y_test[:].values).unsqueeze(1)

# Зададим число термов равное 2 для каждого входного признака
n_terms = [2 for _ in range(n_features)]
# Зададим число выходов
n_out_vars1 = 4
# Зададим шаг обучения
lr = 3e-4
# Зададим тип задачи
task_type1 = "classification"
# Зададим размер подвыборки
batch_size = 64
# Зададим тип функций принадлежности
member_func_type = "gauss"
# Зададим число эпох
epochs = 10
# Флаг, выводить ли информацию в процессе обучения
verbose = True
# На каком устройстве произволить обучение модели ('cpu', 'cuda')
device = "cpu" # "cuda" - обучение будет происходить на гпу

# Создадим модель
model = Model(X_train.iloc[:, 0: n_features].values, Y_train[:].values,
              n_terms, n_out_vars1,
              lr,
              task_type1,
              batch_size,
              member_func_type,
              epochs,
              verbose,
              device=device)

# обучаем моедель
m = model.train()
# Если обучение происходило на ГПУ, то для предсказания модели подаваемые ей данные необходимо также
# перенести на ГПУ (обученная модель и данные для предсказания должны находиться на одном device)

# используем модель, подавая на вход вектор признаков,
# например первого объекта из тестовой выборки, далее определяем ценовую категорию
if model.device.type == "cpu":
    res = m(x[0, :].unsqueeze(0))
else:
    res = m(x[0, :].unsqueeze(0).cuda())
print(res.cpu())
print(torch.argmax(res.cpu(), dim=1))
