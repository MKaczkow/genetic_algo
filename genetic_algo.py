'''
Author: Maciej Kaczkowski
16-24.03.2021
'''


import igraph as ig
import numpy as np
import pandas as pd
import random
import time


def initiate_population(size, vertices_number=25):
    population = np.random.randint(2, size=(size, vertices_number))

    return population


def evaluate(population, graph):
    result = population.shape[1]
    solved = False

    for i in range(population.shape[0]):
        member = population[i, :]
        j = 0
        not_covered = graph.copy()
        to_delete = np.array([])

        for v in not_covered.vs:
            if member[j] == 1:
                to_delete = np.append(to_delete, v)
            j += 1

        not_covered.delete_vertices(to_delete)

        if not_covered.ecount() == 0:
            solved = True

            if np.sum(member) < result:
                result = np.sum(member)

    return solved, result


def tournament_selection(population, graph):
    # przyjmujemy założenie o turnieju 2-osobnikowym
    new_population = np.empty_like(population)

    for i in range(new_population.shape[0]):
        random_numbers = random.sample(range(population.shape[0]), 2)
        first_competitor = population[random_numbers[0]]
        second_competitor = population[random_numbers[1]]

        j = 0
        k = 0
        g1 = graph.copy()
        g2 = graph.copy()
        to_delete_first = np.array([], dtype=int)
        to_delete_second = np.array([], dtype=int)

        for v in g1.vs:
            if first_competitor[j] == 1:
                to_delete_first = np.append(to_delete_first, v.index)
            j += 1

        for v in g2.vs:
            if second_competitor[k] == 1:
                to_delete_second = np.append(to_delete_second, v.index)
            k += 1

        g1.delete_vertices(to_delete_first)
        g2.delete_vertices(to_delete_second)

        if g1.ecount() <= g2.ecount():

            if g1.ecount() < g2.ecount():
                new_population[i:] = first_competitor
            else:
                # if both competitors cover the same number of edges, one with fewer vertices wins
                new_population[i:] = first_competitor if np.sum(first_competitor) < np.sum(second_competitor) \
                    else second_competitor

        else:
            new_population[i:] = second_competitor

    return new_population


def mutate(population, proba_mutation=0.01):
    new_population = population
    random_numbers = np.random.rand(new_population.shape[0], 1)

    if np.all(random_numbers) < proba_mutation: return new_population

    for i in range(random_numbers.shape[0]):
        if random_numbers[i] <= proba_mutation:
            mutated_element = np.random.randint(new_population.shape[1], size=1)
            new_population[i][mutated_element] = abs(new_population[i][mutated_element] - 1)

            if random_numbers[i] <= proba_mutation**2:
                new_population[i][mutated_element - 1] = abs(new_population[i][mutated_element - 1] - 1)

    return new_population


def evolve(population, graph, max_iterations=100, proba_mutation=0.01):
    t0 = time.time()
    generation = 0

    while generation < max_iterations:
        # selekcja turniejowa
        new_population = tournament_selection(population, graph)
        # mutacja
        new_population = mutate(new_population, proba_mutation)
        # sukcesja generacyjna
        population = new_population
        generation += 1

    solved, best_result = evaluate(population, graph)
    t1 = time.time()

    if solved:
        return best_result, t1 - t0
    else:
        return population.shape[1], t1 - t0


def run_tests(population, graph, name):
    proba_mutations = np.array([0.01, 0.05])
    max_iterations = np.array([100, 500])
    results = np.zeros((max_iterations.shape[0] + proba_mutations.shape[0], 2))
    i = 0

    for j in range(max_iterations.shape[0]):
        for k in range(proba_mutations.shape[0]):
            results[i][0], results[i][1] = evolve(population, graph,
                                            max_iterations=max_iterations[j],
                                            proba_mutation=proba_mutations[k])
            i += 1

    pd.DataFrame(results).to_csv('./results_' + name + '.csv', sep='\t')


def show_menu():
    while True:
        choice = input("Select option:\n"
                        "1 - test with full graph\n"
                        "2 - test with bipartite graph\n"
                        "3 - test with random graph\n"
                        "q - quit\n")
        if choice == 'q':
            return None

        try:
            choice = int(choice)
        except ValueError:
            print('Write a number')

        vertices_number = 25
        population_sizes = np.array([10, 50], dtype=int)*vertices_number

        if choice == 1:
            full_graph = ig.Graph.Full(vertices_number)

            for j in population_sizes:
                name = 'full_' + str(j)
                population = initiate_population(size=j)
                run_tests(population=population, graph=full_graph, name=name)

        elif choice == 2:
            edges_number = 2*vertices_number

            n1 = np.array([5, 12])

            for j in n1:

                for k in population_sizes:
                    name = 'bipartite_' + str(j) + '_' + str(k)
                    population = initiate_population(size=k)
                    bipartite_graph = ig.Graph.Random_Bipartite(n1=j, n2=vertices_number - j, m=edges_number)
                    run_tests(population=population, graph=bipartite_graph, name=name)

        elif choice == 3:
            for j in range(2):
                random_graph = ig.Graph.Full(vertices_number)

                for k in range(6 * random_graph.ecount() // 10):
                    which_edge = np.random.randint(random_graph.ecount())
                    edge = random_graph.get_edgelist()[which_edge]
                    random_graph.delete_edges(edge)

                for m in population_sizes:
                    name = 'random_' + str(j) + '_' + str(m)
                    population = initiate_population(size=m)
                    run_tests(population=population, graph=random_graph, name=name)

        else:
            print('Select 1, 2, 3 or q')


# setting random seed using best practice according to numpy docs
rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(123456)))

show_menu()
