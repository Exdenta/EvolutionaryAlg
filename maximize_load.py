import numpy as np


class Item:
    def __init__(self, name, cost, weight):
        self.name = name
        self.cost = cost
        self.weight = weight


def analytical_solution(items: np.array, max_weight: int):
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

    def get_combination_cost_and_weight(row: int):
        weight = 0
        cost = 0
        for col in range(0, item_count):
            if combination_matrix[row, col] == 1:
                weight += items[col].weight
                if weight > max_weight:
                    cost = -1
                    break
                else:
                    cost += items[col].cost
        return cost, weight

    # calculate each combination cost
    combination_costs = np.zeros(combination_count)
    for row in range(0, combination_count):
        combination_costs[row], _ = get_combination_cost_and_weight(row)

    # get best cost result
    best_combination_idx = np.argmax(combination_costs)

    # print
    max_cost, weigth = get_combination_cost_and_weight(best_combination_idx)
    print("combination: ", combination_matrix[best_combination_idx])
    print("combination cost: ", max_cost)
    print("combination wight: ", weigth)


def main():

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
        Item('item13', 73, 3),
        Item('item14', 3000, 300)
    ]

    analytical_solution(items, max_weight)


if __name__ == "__main__":
    main()
