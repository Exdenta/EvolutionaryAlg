"""
* @ author Lex Sherman
* @ email alexandershershakov@gmail.com
* @ create date 2021-08-30 11: 00: 00
* @ modify date 2021-08-30 11: 00: 00
* @ desc Evolutionary algorithm for Knapsack problem
"""

import numpy as np
import random
import time


class Item:
    def __init__(self, name, cost, weight):
        self.name = name
        self.cost = cost
        self.weight = weight


class AnalyticalAlg():

    def __get_combination_cost_and_weight(self, offspring: np.array, items: np.array, max_weight: int) -> tuple:
        """ Get offspring fitness
            Calculated offspring weight and cost

            Args:
                offspring (np.array): input combination of items (array of 0 and 1 - has item in combination or not)
                items (np.array): items description with weight and cost of each
                max_weight (int): maximum weight of combination

            Returns:
                (cost, weight): offspring cost and weight
        """
        weight = 0
        cost = 0
        for col in range(0, len(items)):
            if offspring[col] == 1:
                weight += items[col].weight
                if weight > max_weight:
                    cost = -1
                    break
                else:
                    cost += items[col].cost
        return cost, weight

    def run(self, items: np.array, max_weight: int):
        """ Brute force solution
        """
        item_count = len(items)
        combination_count = pow(2, item_count)
        combination_matrix = np.zeros(
            (combination_count, item_count), dtype=np.uint8)

        # get all combinations
        for col in range(0, combination_count):
            step = pow(2, col)
            for row in range(step, combination_count, 2 * step):
                combination_matrix[row: row + step, col] = 1

        # calculate each combination cost
        combination_costs = np.zeros(combination_count)
        for row in range(0, combination_count):
            combination_costs[row], _ = self.__get_combination_cost_and_weight(
                combination_matrix[row], items, max_weight)

        # get best cost result
        best_combination_idx = np.argmax(combination_costs)

        # print
        max_cost, weigth = self.__get_combination_cost_and_weight(
            combination_matrix[best_combination_idx], items, max_weight)
        print("combination: ", combination_matrix[best_combination_idx])
        print("combination cost: ", max_cost)
        print("combination wight: ", weigth)


class EvolutionalyAlg():

    def __get_combination_cost_and_weight(self, offspring: np.array, items: np.array, max_weight: int) -> tuple:
        """ Get offspring fitness
            Calculated offspring weight and cost

            Args:
                offspring (np.array): input combination of items (array of 0 and 1 - has item in combination or not)
                items (np.array): items description with weight and cost of each
                max_weight (int): maximum weight of combination

            Returns:
                (cost, weight): offspring cost and weight
        """
        weight = 0
        cost = 0
        for col in range(0, len(items)):
            if offspring[col] == 1:
                weight += items[col].weight
                if weight > max_weight:
                    cost = -1
                    break
                else:
                    cost += items[col].cost
        return cost, weight

    def __generate_offspring_rank_selection(self, population: np.array, mutation_chance: float = 0.01):
        """ Rank selection
            the more successfull the offspring the higher rank it gets,
            each offspring count is determined by the rank

            Args:
                population (np.array): initial population sorted by fitness from most fit to less

            Returns:
                new population (np.array): array that is the same size as input population
        """
        item_count = len(population[0])

        # generate offsprings
        population_ranks = np.arange(len(population), 0, -1)
        offspring_count = sum(population_ranks)
        new_population = np.zeros(
            (offspring_count, item_count), dtype=np.uint8)
        counter = 0
        for i in np.arange(0, len(population)):
            new_population[counter: counter +
                           population_ranks[i], :] = population[i]
            counter += population_ranks[i]

        np.random.shuffle(new_population)

        # apply mutation
        for offspring_idx in np.arange(0, len(population)):
            for item_idx in np.arange(0, item_count):
                if random.random() <= mutation_chance:
                    if new_population[offspring_idx][item_idx] == 0:
                        new_population[offspring_idx][item_idx] = 1
                    else:
                        new_population[offspring_idx][item_idx] = 0

        return new_population[0: len(population)]

    def run(self, items: np.array, max_weight: int, max_epoch_count: int, population_size: int, mutation_chance: float = 0.01):
        """ Evolutionary solution
            - generational (replacing entire population each generation)
        """

        # combination_count = pow(2, item_count)
        item_count = len(items)
        population_fitness = np.zeros(population_size, dtype=np.int32)
        population_matrix = np.zeros(
            (population_size, item_count), dtype=np.uint8)

        # generate random population
        for row in range(0, population_size):
            cost = -1
            comb = None
            while (cost < 0):
                comb = np.random.randint(2, size=item_count)
                cost = self.__get_combination_cost_and_weight(
                    comb, items, max_weight)[0]
            population_fitness[row] = cost
            population_matrix[row] = comb

        # sort both arrays
        permutation = population_fitness.argsort()[::-1]
        population_fitness = population_fitness[permutation]
        population_matrix = population_matrix[permutation]

        for i in range(0, max_epoch_count):
            # generate offsprings
            new_population = self.__generate_offspring_rank_selection(
                population_matrix, mutation_chance)

            # calculate population fitness
            for row in range(0, population_size):
                cost, _ = self.__get_combination_cost_and_weight(
                    new_population[row], items, max_weight)
                population_fitness[row] = cost
                population_matrix[row] = new_population[row]

            # sort both arrays
            permutation = population_fitness.argsort()[::-1]
            population_fitness = population_fitness[permutation]
            population_matrix = population_matrix[permutation]

            print(f"best after {i} epoch: cost: " + str(
                population_fitness[0]) + ", combination: " + str(population_matrix[0]))

        print(population_fitness)


def main():

    mutation_chance = 0.02

    # population size for evolutionary alg
    population_size = 1000

    # maximum epoch count
    max_epoch_count = 30

    # maximum load in the bag
    max_weight = 1000

    # all possible items we can take
    items: np.array = [
        Item('item1', 100, 110),
        Item('item2', 200, 220),
        Item('item3', 300, 330),
        Item('item4', 400, 440),
        Item('item5', 500, 550),
        Item('item6', 600, 660),
        Item('item7', 700, 770),
        Item('item8', 800, 880),
        Item('item9', 900, 990),
        Item('item10', 3, 53),
        Item('item11', 3, 33),
        Item('item12', 33, 3),
        Item('item13', 313, 3),
        Item('item14', 83, 3),
        Item('item15', 73, 3),
        Item('item16', 3000, 300),
        Item('item17', 31, 123),
        Item('item18', 36, 40),
        Item('item19', 31, 31),
        Item('item20', 319, 7),
    ]

    analytical_solution = AnalyticalAlg()
    evolutionary_solution = EvolutionalyAlg()

    tic = time.perf_counter()
    evolutionary_solution.run(
        items, max_weight, max_epoch_count, population_size, mutation_chance)
    toc = time.perf_counter()
    print(f"Evolutionary: {toc - tic:0.4f} seconds")

    # tic = time.perf_counter()
    # analytical_solution.run(items, max_weight)
    # toc = time.perf_counter()
    # print(f"Brute force: {toc - tic:0.4f} seconds")


if __name__ == "__main__":
    # population = np.arange(100, 0, -1, dtype=np.float32)
    # new_population = generate_offspring_rank_selection(population)
    # new_population = np.sort(new_population)
    # print(new_population)

    # h = [len(list(group))
    #      for key, group in groupby(new_population)]
    # plt.plot(np.arange(0, len(h)), h)
    # plt.show()

    # # h = np.histogram(new_population)

    # # _ = plt.hist(h, bins='auto')
    # # plt.show()
    main()
