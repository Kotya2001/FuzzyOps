import pandas as pd
from fuzzyops.fuzzy_nn import Model
from sklearn.preprocessing import LabelEncoder
import torch

# Данные для обучения модели ANFIS
classification_data = pd.read_csv("Iris.csv")
# Берем только 2 первых признака (можно и больше)
n_features = 2
# Определяем количестов термов для каждого признака (Слой фаззификации строит столько нечетких чисел)
n_terms = [5, 5]
# Задаем число выходных переменных (в нашем случае предсказываем 3 класс, значит 3 выходные переменные)
n_out_vars1 = 3
# Шаг обучения
lr = 3e-4
# Тип задачи
task_type1 = "classification"
# размер подвыборки для обучения
batch_size = 2
# Тип функции принадлежности ('gauss' - гауссовская, 'bell' - обобщенный колокол)
member_func_type = "gauss"
# Число итераций
epochs = 100
# Флаг, выводить ли информацию в процессе обучения
verbose = True
# На каком устройстве произволить обучение модели ('cpu', 'cuda')
device = "cpu" # "cuda" - обучение будет происходить на гпу

# Данные
X_class, y_class = classification_data.iloc[:, 1: 1 + n_features].values, \
                             classification_data.iloc[:, -1]

# Кодируем целевую переменную, так как она представлена строковым типом
le = LabelEncoder()
y = le.fit_transform(y_class)

# инициализируем модель
model = Model(X_class, y,
              n_terms, n_out_vars1,
              lr,
              task_type1,
              batch_size,
              member_func_type,
              epochs,
              verbose,
              device=device)

# создание экземпляра класса
m = model.train()
# предсказание Если обучение происходило на ГПУ, то для предсказания модели подаваемые ей данные необходимо также
# перенести на ГПУ (модель и данные для предсказания должны находиться на одном device)
if model.device.type == "cpu":
    res = m(torch.Tensor([[5.1, 3.5]]))
else:
    res = m(torch.Tensor([[5.1, 3.5]]).cuda())
print(res.cpu())
print(torch.argmax(res.cpu(), dim=1))