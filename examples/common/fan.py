from fuzzyops.fuzzy_numbers import Domain
from fuzzyops.fan import Graph, calc_final_scores

# Создание домена для построения оценко (степеней осуществимости работ,
# задаваемых экспертами - оценки также можно задавать с помощью четких чисел)
# вершина - это какой-либо этап выполениия проекта в нечеткой сетевой моделе
score_domain = Domain((0, 1, 0.01), name='scores')

# имеется два критерия для оценки каждого этапа (сложность, наличие ресурсов, поэтому необходиом оценить итоговую
# оценку)
score_domain.create_number('triangular', 0.4, 0.7, 0.9, name='complex_A')
score_domain.create_number('triangular', 0.4, 0.76, 1, name='sources_A')
A_score = calc_final_scores([score_domain.complex_A, score_domain.sources_A])

score_domain.create_number('triangular', 0.5, 0.7, 0.95, name='complex_A2')
score_domain.create_number('triangular', 0.4, 0.85, 1, name='sources_A2')
A2_score = calc_final_scores([score_domain.complex_A2, score_domain.sources_A2])

score_domain.create_number('triangular', 0.5, 0.77, 1, name='complex_B')
score_domain.create_number('triangular', 0.44, 0.89, 1, name='sources_B')
B_score = calc_final_scores([score_domain.complex_B, score_domain.sources_B])

score_domain.create_number('triangular', 0.3, 0.6, 0.91, name='complex_C')
score_domain.create_number('triangular', 0.4, 0.7, 1, name='sources_C')
C_score = calc_final_scores([score_domain.complex_C, score_domain.sources_C])

score_domain.create_number('triangular', 0.4, 0.61, 0.81, name='complex_D')
score_domain.create_number('triangular', 0.43, 0.7, 1, name='sources_D')
D_score = calc_final_scores([score_domain.complex_D, score_domain.sources_D])

score_domain.create_number('triangular', 0.5, 0.6, 0.91, name='complex_E')
score_domain.create_number('triangular', 0.5, 0.8, 1, name='sources_E')
E_score = calc_final_scores([score_domain.complex_E, score_domain.sources_E])

# строим нечеткую аналитическую сеть (последовательность выполенения работ) и задаем каждому ребру степень
# осуществимости как максимум из степеней осуществимости вершин
graph = Graph()
graph.add_edge('Start', 'A', A_score)
graph.add_edge('Start', "A2", A2_score)
graph.add_edge('A', 'B', max(A_score, B_score))
graph.add_edge("A2", 'B', max(A2_score, B_score))
graph.add_edge('B', 'C', max(C_score, B_score))
graph.add_edge('C', 'D', max(C_score, D_score))
graph.add_edge('C', 'E', max(C_score, E_score))
graph.add_edge('D', 'End', D_score)
graph.add_edge('E', 'End', E_score)


# Находим наиболее осуществимый путь выполенния проекта (Наиболее предпочтительную последовательность алтернатив)
most_feasible_path = graph.find_most_feasible_path('Start', 'End')
if most_feasible_path:
    print(f"Most feasible path: {most_feasible_path}")
    print(f"Feasibility: {graph.calculate_path_fuzziness(most_feasible_path)}")
else:
    print("No path found.")
