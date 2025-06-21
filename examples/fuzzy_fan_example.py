"""
Task:
Suppose an investment company is planning to develop a digital investment advisor
(a kind of personal recommendation system for making investment decisions)

The development of such a system involves several stages:

    Market research;
    System design;
    Product development and testing;
    System launch;

At the same time, each stage can be performed in different ways (alternatives),
which differ from each other in terms of criteria (for example, cost, execution time, complexity, etc.).
It is necessary to choose the most feasible way of completing the project, taking into account the uncertainty and vagueness in the estimates
of parameters for each stage.

It is proposed to solve the problem using a fuzzy alternative network model of project management.

Consider the stages of the project:

    1. Market research:
        Alternative 1: Conducting surveys, questionnaires among the company's users or among other people
        (the cost is likely to be "low", but the time required is "medium") (essentially collecting data from your customers);
        Alternative 2: Searching for (buying) existing data (pre-existing survey databases) (the cost is "medium", and the time required is "quick");

    2. Designing:
        Alternative 1: Designing your own technologies for the project implementation (for example, developing your own
        Decision Support System (architecture) for a digital consultant, your own recommendation system algorithms, your own DBMS, etc.) (cost "high", time "long");
        Alternative 2: Using ready-made technologies for the project implementation (for example, purchasing similar DSS
        and slightly modifying them, using widely available recommendation system algorithms, etc.) (cost "low", time "quick")

    3. Development and testing:
        Alternative 1: Internal development (in-house resource development) (Cost is "average", execution time may be "long");
        Alternative 2: Third-party development (outsourcing) (Cost is "low", execution time is "fast");

    4. Product Release:
        Alternative 1: Careful preparation of the product presentation, organization of events
        (cost is "high", time is "long");
        Alternative 2: Product launch and small advertising (cost is "low", time is "fast")

Next, you need to build an analytical network,
add edges to it that correspond to the work and assign them a weight.

The example shows how you can set weights using fuzzy numbers.
Weights can also be obtained in a different way when evaluating each alternative according to criteria.
The example uses two criteria - the cost of the work and its time of discharge (hours)

Let's say there is data that indicates the cost and time of work for each alternative in stages:
It is necessary to find the degrees of confidence of the values of the source data for the corresponding fuzzy numbers,
then multiply the values of the degrees of confidence for the criteria time and cost
and assign the resulting numbers as the weight of the edge in the graph

(For example, for alternative 1 in the Research stage, we find the degrees of confidence
for the cost and completion time of the constructed fuzzy numbers, multiply them, and set them as the weight for the edge
of alternative 1 in the Research stage)

As a result, it is necessary to build a graph and find the most feasible path in

"""

from fuzzyops.fan import Graph, calc_final_scores
from fuzzyops.fuzzy_numbers import Domain, FuzzyNumber

# Possible input data
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

# the boundaries and step for the domain and the domains themselves are set depending
# on the project budget for specific stages and the expert assessment

# domain data cost in $, for the first stage - "Research"
cost_survey_domain = Domain((0, 10000, 100), name='cost_survey')
# in a house for setting the time for performing any type of work in hours (640 hours is the maximum, for example)
time_domain = Domain((0, 640, 1), name='time')

# domain the cost of designing new technologies and/or using existing ones in $,
# for the second stage, "Design" (you can create one domain,
# to which you can then add the cost of the various stages with their terms).
# We also use this domain to define terms based on the 3rd stage
cost_design_domain = Domain((0, 100000, 100), name='cost_design')

# low cost for alternative 1 in the "Research" stage
not_high_survey_cost = cost_survey_domain.create_number("trapezoidal", 0, 500, 1500, 2000, name="not_high_survey_cost")
# the average cost for alternative 2 in the "Research" stage
middle_survey_cost = cost_survey_domain.create_number("trapezoidal", 1800, 2700, 3900, 4800, name="middle_survey_cost")
# average work time for alternative 1 in the Research stage"
middle_time_survey = time_domain.create_number("trapezoidal", 16, 24, 32, 40, name="middle_time_survey")
# fast search time for existing data in alternative 2 in the Research phase
high_time_survey = time_domain.create_number("trapezoidal", 5, 9, 15, 23, name="high_time_survey")

# estimates for alternative 1 in the "Research" phase
score_research_1 = not_high_survey_cost(data["not_high_survey_cost"]).item() \
                   * middle_time_survey(data["middle_time_survey"]).item()
# estimates for alternative 2 in the "Research" phase
score_research_2 = middle_survey_cost(data["middle_survey_cost"]).item() \
                   * high_time_survey(data["high_time_survey"]).item()

# high cost for alternative 1 in the "Design" phase
high_design_cost = cost_design_domain.create_number("trapezoidal", 30000, 40000, 100000, 100000,
                                                    name="high_design_cost")
# low cost for alternative 2 in the "Design" stage"
not_high_design_cost = cost_design_domain.create_number("trapezoidal", 0, 1000, 4000, 10000,
                                                        name="not_high_design_cost")
# fast time development of alternative 2 in the "Design" stage
not_high_time_design = time_domain.create_number("trapezoidal", 40, 120, 160, 200, name="not_high_time_design")
# long development time for alternative 1 in the "Design" stage
high_time_design = time_domain.create_number("trapezoidal", 120, 280, 640, 640, name="high_time_design")

# estimates for alternatives 1 and 2 in the "Design" stage
score_design_1 = high_design_cost(data["high_design_cost"]).item() \
                 * high_time_design(data["high_time_design"]).item()
score_design_2 = not_high_design_cost(data["not_high_design_cost"]).item() \
                 * not_high_time_design(data["not_high_time_design"]).item()

# average cost for alternative 1 in the "Development" phase
middle_dev_cost = cost_design_domain.create_number("trapezoidal", 40000, 60000, 100000, 100000, name="middle_dev_cost")
# low cost for alternative 2 in the "Development" stage
not_high_dev_cost = cost_design_domain.create_number("trapezoidal", 10000, 25000, 40000, 55000,
                                                     name="not_high_dev_cost")
# fast development time for alternative 2 in the Development stage"
not_high_time_dev = time_domain.create_number("trapezoidal", 90, 170, 200, 240, name="not_high_time_dev")
# long development time for alternative 1 in the Development phase"
high_time_dev = time_domain.create_number("trapezoidal", 160, 320, 640, 640, name="high_time_dev")

# estimates for alternatives 1 and 2 in the "Development" stage
score_dev_1 = middle_dev_cost(data["middle_dev_cost"]).item() \
              * high_time_dev(data["high_time_dev"]).item()
score_dev_2 = not_high_dev_cost(data["not_high_dev_cost"]).item() \
              * not_high_time_dev(data["not_high_time_dev"]).item()

# high cost for alternative 1 in the "Product Release" stage
high_prod_cost = cost_design_domain.create_number("trapezoidal", 40000, 55000, 100000, 100000, name="high_prod_cost")
# low cost for alternative 2 in the "Product Release" stage
not_high_prod_cost = cost_design_domain.create_number("trapezoidal", 7000, 20000, 30000, 40000,
                                                      name="not_high_prod_cost")
# fast development time for alternative 2 in the Product Release stage
not_high_time_prod = time_domain.create_number("trapezoidal", 30, 50, 90, 140, name="not_high_time_prod")
# долгое вермя разработки по альтернативе 1 в этапе "Выпуск продукта"
high_time_prod = time_domain.create_number("trapezoidal", 120, 200, 640, 640, name="high_time_prod")

# estimates for alternatives 1 and 2 in the "Product Release" stage
score_prod_1 = high_prod_cost(data["high_prod_cost"]).item() \
               * high_time_prod(data["high_time_prod"]).item()
score_prod_2 = not_high_prod_cost(data["not_high_prod_cost"]).item() \
               * not_high_time_prod(data["not_high_time_prod"]).item()

# Building a fuzzy analytical network

# Creating a graph
graph = Graph()

# Adding edges with fuzzy estimates
graph.add_edge("Start", "Research1", score_research_1)  # Alternative 1 for research
graph.add_edge("Start", "Research2", score_research_2)  # Alternative 2 for research

graph.add_edge("Research1", "Design1", max(score_research_1, score_design_1))  # Alternative 1 for designing
graph.add_edge("Research1", "Design2", max(score_design_2, score_research_1))  # Alternative 1 for designing

graph.add_edge("Research2", "Design1", max(score_design_1, score_research_2))  # Alternative 2 for design
graph.add_edge("Research2", "Design2", max(score_design_2, score_research_2))  # Alternative 2 for design

graph.add_edge("Design1", "Dev1", max(score_dev_1, score_design_1))  # Alternative 1 for development
graph.add_edge("Design1", "Dev2", max(score_dev_2, score_design_1))  # Alternative 1 for development

graph.add_edge("Design2", "Dev1", max(score_dev_1, score_design_2))  # Alternative 2 for development
graph.add_edge("Design2", "Dev2", max(score_dev_2, score_design_2))  # Alternative 2 for development

graph.add_edge("Dev1", "Production1", max(score_prod_1, score_dev_1))  # Alternative 1 for product release
graph.add_edge("Dev1", "Production2", max(score_prod_2, score_dev_1))  # Alternative 1 for product release

graph.add_edge("Dev2", "Production1", max(score_prod_1, score_dev_2))  # Alternative 2 for product release
graph.add_edge("Dev2", "Production2", max(score_prod_2, score_dev_2))  # Alternative 2 for product release

graph.add_edge("Production1", "End", score_prod_1)  # Completion
graph.add_edge("Production1", "End", score_prod_2)  # Completion

# Finding the most feasible way
most_feasible_path = graph.find_most_feasible_path("Start", "End")
print("The most feasible way:", most_feasible_path)

# We use a macro algorithm to select the best alternative
best_alternative, max_feasibility = graph.macro_algorithm_for_best_alternative()
print("The best alternative:", best_alternative)
print("Feasibility assessment:", max_feasibility)
