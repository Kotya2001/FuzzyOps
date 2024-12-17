"""
Задача:
    Необходимо смоделировать вероятность проникновения злоумышленников в корпоративную систему, а также
    вероятность рапспрстарнения атаки и уровень риска при помощи построения базы правил и нечеткого логического вывода

Нечеткие переменные:
    X1 - Число активных пользователей в системе
    X2 - Время (часы)
    Y1 - Восможность проникновения
    Y2 - Возможность распространения атаки
    Y3 - Уровень риска

База правил:
    Если пользователей (Х1) "несколько" И время (Х2) "нерабочее", То Возможность проникновения (Y1) "средняя";
    Если пользователей (Х1) "много" И время (Х2) "рабочее", То Возможность проникновения (Y1) "низкая";
    Если пользователей (Х1) "много" И время (Х2) "нерабочее", То Возможность проникновения (Y1) "высокая";

    Если пользователей (Х1) "несколько" И время (Х2) "нерабочее", То Возможность распространения атаки (Y2) "средняя";
    Если пользователей (Х1) "много" И время (Х2) "рабочее", То Возможность распространения атаки (Y2) "низкая";
    Если пользователей (Х1) "много" И время (Х2) "несколько", То Возможность распространения атаки (Y2) "высокая";

    Если Возможность проникновения (Y1) "низкая", То уровень риска (Y3) "низкий";
    Если Возможность проникновения (Y1) "средняя", То уровень риска (Y3) "средний";
    Если Возможность проникновения (Y1) "высокая", То уровень риска (Y3) "высокий";
"""

# Импорт необходимых классов для построения нечеткого логического вывода по алгоритму Мамдини
# (Библиотека уже установлена в ваш проект)
from fuzzyops.fuzzy_logic import BaseRule, FuzzyInference
from fuzzyops.fuzzy_numbers import Domain, FuzzyNumber


# База правил
rules = [
    BaseRule(
        antecedents=[('user', 'several'), ('time', 'not_work_time')],
        consequent=('attack', 'middle'),
    ),
    BaseRule(
        antecedents=[('user', 'many'), ('time', 'work_time')],
        consequent=('attack', 'low'),
    ),
    BaseRule(
        antecedents=[('user', 'many'), ('time', 'not_work_time')],
        consequent=('attack', 'high'),
    ),
    BaseRule(
        antecedents=[('user', 'several'), ('time', 'not_work_time')],
        consequent=('spread', 'middle'),
    ),
    BaseRule(
        antecedents=[('user', 'many'), ('time', 'work_time')],
        consequent=('spread', 'low'),
    ),
    BaseRule(
        antecedents=[('user', 'many'), ('time', 'not_work_time')],
        consequent=('spread', 'high'),
    ),
    BaseRule(
        antecedents=[('attack', 'low')],
        consequent=('risk', 'low'),
    ),
    BaseRule(
        antecedents=[('attack', 'middle')],
        consequent=('risk', 'middle'),
    ),
    BaseRule(
        antecedents=[('attack', 'high')],
        consequent=('risk', 'high'),
    )
]

# X1
user_domain = Domain((0, 30), name='user')
user_domain.create_number('trapezoidal', -1, 0, 4, 7, name='several')
# создание терма "много" пользователей
many_users = user_domain.get('several').negation
user_domain.many = many_users

# X2
time_domain = Domain((0, 24), name='time')
time_domain.create_number('trapezoidal', 8, 9, 18, 19, name='work_time')
# # создание терма "нерабочее" время
not_work_time = time_domain.get('work_time').negation
time_domain.not_work_time = not_work_time

# Y1
# Нечеткие термы для вероятностей атаки
attack_prob = Domain((0, 1, 0.01), name='attack')
attack_prob.create_number('trapezoidal', -0.1, 0., 0.1, 0.3, name='low')
attack_prob.create_number('trapezoidal', 0.3, 0.43, 0.6, 0.7, name='middle')
attack_prob.create_number('trapezoidal', 0.65, 0.8, 0.9, 1, name='high')

# Y2
# Нечеткие термы для вероятностей распространения
spread_prob = Domain((0, 1, 0.01), name='spread')
spread_prob.create_number('trapezoidal', -0.1, 0., 0.15, 0.34, name='low')
spread_prob.create_number('trapezoidal', 0.32, 0.4, 0.5, 0.7, name='middle')
spread_prob.create_number('trapezoidal', 0.64, 0.76, 0.9, 1, name='high')

# Y3
# Нечеткие термы для шкалы риска (к примеру от 0 до 10)
risk = Domain((0, 10), name='risk')
risk.create_number('trapezoidal', -1, 0., 2, 3, name='low')
risk.create_number('trapezoidal', 3, 4, 5, 6, name='middle')
risk.create_number('trapezoidal', 6, 8, 10, 10, name='high')

inference_system = FuzzyInference(domains={
    'user': user_domain,
    'time': time_domain,
    'attack': attack_prob,
    'spread': spread_prob,
    'risk': risk
}, rules=rules)

input_data = {
    'user': 35,  # много людей еще на работе
    'time': 17  # 17:00
}

result = inference_system.compute(input_data)
print(result)
