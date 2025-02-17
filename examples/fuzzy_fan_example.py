"""
Задача:
Предположим инвестиционная компания планирует разработать цифрового инвестиционного консультанта
(подобие персональной рекомендательной системы для принятия инвестиционных решений)

Разработка такой системы включает в себя несколько этапов:

    Исследование рынка;
    Проектирование системы;
    Разраотка продукта;
    Тестирование продукта;
    Запуск системы в эксплуатацию;

При этом для для каждого этапа может быть выполнен различными способами (альтернативами),
которые отличаются друг от друга по критериями (например, стоимость, время выполнения, сложность, и.т.д).
Необходимо выбрать наиболее осуществимый путь выполнения проекта, учитывая неопределенность и нечеткость в оценках
параметров для каждого этапа

Предлагается решить задачу с помощью нечеткой альтернативной сетевой модели управления проектами.

Рассмотрим этапы проекта:

    1. Исследование рынка:
        Альтернатива 1: Проведение опросов анкетирование среди пользователей компании или среди других людей
        (стоимость скорее всего "недорогая", но время на реализацию "среднее") (по сути сбор данных со своих клиентов);
        Альтернатива 2: Поиск(покупка) существующих данных (уже готовые базы опросов) (стоимость "средняя", а время на
        получение "быстрое");

    2. Проектирование:
        Альтернатива 1: Проектирование собственных технологий для реализации проекта (например, разработка своей
        Системы поддержки принятия решения (архитектуры) для цифрового консультанта,
        своих алгоритмов рекомендательных систем, своих СУБД, и.т.д) (стоимость "высокая", время "долгое");
        Альтернатива 2: Использование уже готовых технологий для реализации проекта (например, покупка похожих СПР
        и небольшая их доработка, использование например широкораспространенных алгоритмов рекомендательных системы,
        вообщем использование лучших практик) (стоимость "низкая", затраченное время на этап "быстрое")

    3. Разработка и тестирование:
        Альтернатива 1: Внутренняя разработка (разработка собственными ресурсами) (Стоимость "средняя",
        время выполнения может быть "долгое");
        Альтернатива 2: Разработка с помощью сторонних копманий (аутсорсинг) (Стоимость "низкая",
        время выполнения "быстрое");

    4. Выпуск продукта:
        Альтернатива 1: Тщательная подготовка презентации продукта, организация мероприятий
        (стоимость "высокая", время "долгое");
        Альтернатива 2: Запуск продукта и небольшая реклама (стоимость "низкая", время "быстрое")

Далее необходимо построить аналитическую сеть,
добавить в нее ребра, которые соответствуюут работам и назначить им вес.

В примере рассмотрено, как можно задать веса с помощью нечетких чисел.
Веса также могут быть получены другим путем при оценке каждой альтернативы по критериям.
В примере используется два критерия - стоимость работ и их время выоплнения (часы)

Пусть имеются данные, которые означают стоимость и время выполнения работ для каждой альтернативы в этапах:
Необходимо, найти степени уверенности значений исходных данных у соответствующих им нечетких чисел,
затем перемножить значения степеней уверенности у критериев время и стоимость
и назначить полученное числа как вес ребра в графе

(Например, для альтернативы 1 в этапе Исследование находим степени уверенности
для стоимости и времения выполенния у построенных нечетких чисел, перемножаем их, и задаем как вес для ребра
альтернативы 1 в этапе Исследование)

В итоге, необходимо построить граф и найти наиболее осуществимый путь выполнения проекта
(последовательность вершин) и оценку осуществимости

"""


from fuzzyops.fan import Graph, calc_final_scores
from fuzzyops.fuzzy_numbers import Domain, FuzzyNumber

# Возможные входные данные
data = {
    "not_high_survey_cost": 899,
    "middle_survey_cost": 3100,
    "middle_time_survey": 25,
    "high_time_survey": 16,
    "high_design_cost": 86000,
    "not_high_design_cost": 4300,
    "not_high_time_design": 90,
    "high_time_design": 300,
    "middle_dev_cost": 56000,
    "not_high_dev_cost": 34000,
    "not_high_time_dev": 87,
    "high_time_dev": 270,
    "high_prod_cost": 67000,
    "not_high_prod_cost": 19000,
    "not_high_time_prod": 48,
    "high_time_prod": 150
}

# границы и шаг для домена и сами доменты задаются в зависимости
# от бюджета проекта на конкретные этапы и экспертной оценки ЛПР

# домен стоимость данных в $, для первого этапа - "Исследование"
cost_survey_domain = Domain((0, 10000, 100), name='cost_survey')
# доме для задания времени выполнения какого-либо типа работ в часах (640 часов максимальное, например)
time_domain = Domain((0, 640, 1), name='time')

# домен стоимость проектирования новых технологи/или использование существующих в $,
# для второго этапа - "Проектирование" (можно создать один домен,
# в который потом можно добавлять стоимость различных этапов с их термами).
# Используем также это домен для задания термов на альтернативы 3-его этапа
cost_design_domain = Domain((0, 100000, 100), name='cost_design')

# невысокая стоимость для альтернативы 1 в этапе "Исследование"
not_high_survey_cost = cost_survey_domain.create_number("trapezoidal", 0, 500, 1500, 2000, name="not_high_survey_cost")
# средняя стоимость для альтернативы 2 в этапе "Исследование"
middle_survey_cost = cost_survey_domain.create_number("trapezoidal", 1800, 2700, 3900, 4800, name="middle_survey_cost")
# среднее вермя работы по альтернативе 1 в этапе Исследование"
middle_time_survey = time_domain.create_number("trapezoidal", 16, 24, 32, 40, name="middle_time_survey")
# быстрое вермя поиска уже существующих данных по альтернативе 2 в этапе Исследование"
high_time_survey = time_domain.create_number("trapezoidal", 5, 9, 15, 23, name="high_time_survey")

# оценки для альтернативы 1 и 2 в этапе "Исследование"
score_research_1 = not_high_survey_cost(data["not_high_survey_cost"]).item() \
                   * middle_time_survey(data["middle_time_survey"]).item()
score_research_2 = middle_survey_cost(data["middle_survey_cost"]).item() \
                   * high_time_survey(data["high_time_survey"]).item()

# высокая стоимость для альтернативы 1 в этапе "Проектирование"
high_design_cost = cost_design_domain.create_number("trapezoidal", 30000, 40000, 100000, 100000,
                                                    name="high_design_cost")
# невысокая стоимость для альтернативы 2 в этапе "Проектирование"
not_high_design_cost = cost_design_domain.create_number("trapezoidal", 0, 1000, 4000, 10000,
                                                        name="not_high_design_cost")
# быстрое вермя разработка по альтернативе 2 в этапе "Проектирование"
not_high_time_design = time_domain.create_number("trapezoidal", 40, 120, 160, 200, name="not_high_time_design")
# долгое вермя разработки по альтернативе 1 в этапе "Проектирование"
high_time_design = time_domain.create_number("trapezoidal", 120, 280, 640, 640, name="high_time_design")

# оценки для альтернативы 1 и 2 в этапе "Проектирование"
score_design_1 = high_design_cost(data["high_design_cost"]).item() \
                 * high_time_design(data["high_time_design"]).item()
score_design_2 = not_high_design_cost(data["not_high_design_cost"]).item() \
                 * not_high_time_design(data["not_high_time_design"]).item()

# средняя стоимость для альтернативы 1 в этапе "Разработка"
middle_dev_cost = cost_design_domain.create_number("trapezoidal", 40000, 60000, 100000, 100000, name="middle_dev_cost")
# невысокая стоимость для альтернативы 2 в этапе "Разработка"
not_high_dev_cost = cost_design_domain.create_number("trapezoidal", 10000, 25000, 40000, 55000,
                                                     name="not_high_dev_cost")
# быстрое вермя разработка по альтернативе 2 в этапе Разработка"
not_high_time_dev = time_domain.create_number("trapezoidal", 90, 170, 200, 240, name="not_high_time_dev")
# долгое вермя разработки по альтернативе 1 в этапе Разработка"
high_time_dev = time_domain.create_number("trapezoidal", 160, 320, 640, 640, name="high_time_dev")

# оценки для альтернативы 1 и 2 в этапе "Разработка"
score_dev_1 = middle_dev_cost(data["middle_dev_cost"]).item() \
              * high_time_dev(data["high_time_dev"]).item()
score_dev_2 = not_high_dev_cost(data["not_high_dev_cost"]).item() \
              * not_high_time_dev(data["not_high_time_dev"]).item()

# высокая стоимость для альтернативы 1 в этапе "Выпуск продукта"
high_prod_cost = cost_design_domain.create_number("trapezoidal", 40000, 55000, 100000, 100000, name="high_prod_cost")
# невысокая стоимость для альтернативы 2 в этапе "Выпуск продукта"
not_high_prod_cost = cost_design_domain.create_number("trapezoidal", 7000, 20000, 30000, 40000,
                                                      name="not_high_prod_cost")
# быстрое вермя разработка по альтернативе 2 в этапе Выпуск продукта"
not_high_time_prod = time_domain.create_number("trapezoidal", 30, 50, 90, 140, name="not_high_time_prod")
# долгое вермя разработки по альтернативе 1 в этапе "Выпуск продукта"
high_time_prod = time_domain.create_number("trapezoidal", 120, 200, 640, 640, name="high_time_prod")

# оценки для альтернативы 1 и 2 в этапе " Выпуск продукта"
score_prod_1 = high_prod_cost(data["high_prod_cost"]).item() \
               * high_time_prod(data["high_time_prod"]).item()
score_prod_2 = not_high_prod_cost(data["not_high_prod_cost"]).item() \
               * not_high_time_prod(data["not_high_time_prod"]).item()

# Создаем граф
graph = Graph()

# Добавляем ребра с нечеткими оценками
graph.add_edge("Start", "Research1", score_research_1)  # Альтернатива 1 для исследования
graph.add_edge("Start", "Research2", score_research_2)  # Альтернатива 2 для исследования

graph.add_edge("Research1", "Design1", max(score_research_1, score_design_1))  # Альтернатива 2 для исследования
graph.add_edge("Research1", "Design2", max(score_design_2, score_research_1))  # Альтернатива 2 для исследования

graph.add_edge("Research2", "Design1", max(score_design_1, score_research_2))  # Альтернатива 2 для исследования
graph.add_edge("Research2", "Design2", max(score_design_2, score_research_2))  # Альтернатива 2 для исследования

graph.add_edge("Design1", "Dev1", max(score_dev_1, score_design_1))  # Альтернатива 2 для исследования
graph.add_edge("Design1", "Dev2", max(score_dev_2, score_design_1))  # Альтернатива 2 для исследования

graph.add_edge("Design2", "Dev1", max(score_dev_1, score_design_2))  # Альтернатива 2 для исследования
graph.add_edge("Design2", "Dev2", max(score_dev_2, score_design_2))  # Альтернатива 2 для исследования

graph.add_edge("Dev1", "Production1", max(score_prod_1, score_dev_1))  # Альтернатива 2 для исследования
graph.add_edge("Dev1", "Production2", max(score_prod_2, score_dev_1))  # Альтернатива 2 для исследования

graph.add_edge("Dev2", "Production1", max(score_prod_1, score_dev_2))  # Альтернатива 2 для исследования
graph.add_edge("Dev2", "Production2", max(score_prod_2, score_dev_2))  # Альтернатива 2 для исследования

graph.add_edge("Production1", "End", score_prod_1)  # Альтернатива 2 для исследования
graph.add_edge("Production1", "End", score_prod_2)  # Альтернатива 2 для исследования

# Находим наиболее осуществимый путь
most_feasible_path = graph.find_most_feasible_path("Start", "End")
print("Наиболее осуществимый путь:", most_feasible_path)

# Выполняем макроалгоритм для выбора наилучшей альтернативы
best_alternative, max_feasibility = graph.macro_algorithm_for_best_alternative()
print("Наилучшая альтернатива:", best_alternative)
print("Оценка осуществимости:", max_feasibility)
