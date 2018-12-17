import numpy as np

""" This is a implementation of HCO algorithm with python
    The test objective function is sphere
    numpy is used to do some linear calculation
"""

#  parameter setting
ld = -10  # low boundary
ud = 10  # up  boundary
dim = 10  # the num of parameter should be optimized
pop_size = 20  # population size

max_evaluation = 20000
run_time = 2
max_flow = 3  # the max count of flow
p_eva = 0.2  # evaporation probability
best_record = []


def hco():
    for r in range(1, run_time):
        evaluation_now = 0
        pop_pos = np.random.randint(ld, ud, size=[pop_size, dim])  # population position
        pop_pos = pop_pos.astype(float)
        fitness = np.zeros(pop_size, dtype=float)  # initialize and calculate the fitness
        calc_fitness(pop_pos, fitness)
        best_index = np.argmin(fitness)  # get the index of the best performance individual
        evaluation_now += pop_size

        while evaluation_now <= max_evaluation:
            best_fit = fitness[best_index]
            best_record.append(best_fit)
            # execute the flow operation
            evaluation_now = flow(pop_pos, fitness, best_index, evaluation_now)
            # update the best index and position of population
            # execute the infiltration operation
            evaluation_now = infiltration(pop_pos, fitness, evaluation_now)
            best_index = np.argmin(fitness)
            # execute the evaporation and precipitation operation
            evaluation_now = eva_and_precip(pop_pos, fitness, best_index, evaluation_now)
            best_index = np.argmin(fitness)
    print(best_record)


def calc_fitness(pop_pos, fitness):
    # this function completes the flow operation
    # Args: pop_pos: population position, individual should be a row vector
    # return: fitness, a numpy array with [row,1] where row is the row of pop_pos
    # calculate the fitness for every individual
    for index, idv in enumerate(pop_pos):
        # fitness[index] = objfun.train(idv)
        fitness[index] = sphere(idv)
    return fitness


def flow(pop_pos, fitness, best_index, evaluation_now):
    # this function completes the flow operation
    # Args: pop_pos: population position, individual should be a row vector
    # fitness: the fitness value of population
    # best_index: the best individual position index
    # evaluation_now: global control parameter
    row, col = np.shape(pop_pos)
    for index, idv in enumerate(pop_pos):
        # randomly choose a direction to flow, should not the same as itself
        flow_time = 0
        while flow_time < max_flow:
            if fitness[index] == fitness[best_index]:
                flow_direction = np.random.randint(0, row)
                while flow_direction == index:
                    flow_direction = np.random.randint(0, row)
            else:
                better_index = np.where(fitness < fitness[index])
                flow_direction = better_index[0][np.random.randint(0, np.size(better_index[0]))]
            # calculate new individual and its fitness
            new_idv = idv + np.multiply(np.random.rand(col),
                                        np.subtract(pop_pos[flow_direction], idv))
            bndry_proce(new_idv)
            # new_fitness = objfun.train(new_idv)
            new_fitness = sphere(new_idv)
            evaluation_now += 1
            # only update the fitness when the new individual perform better than the old one
            if new_fitness < fitness[index]:
                pop_pos[index] = new_idv
                fitness[index] = new_fitness
                best_index = np.argmin(fitness)
                flow_time += 1
            else:
                flow_time = max_flow
            # update the control parameter
    return evaluation_now


def infiltration(pop_pos, fitness, evaluation_now):
    # this function completes the infiltration operation
    # Args: pop_pos: population position, individual should be a row vector
    # fitness: the fitness value of population
    # evaluation_now: global control parameter
    row, col = np.shape(pop_pos)
    for index, idv in enumerate(pop_pos):
        # A randomly chosen solution is used in producing a mutant solution of the solution
        # and should not be the same
        ngh_index = np.random.randint(0, row)  # neighbour index
        while ngh_index == index:
            ngh_index = np.random.randint(0, row)
        neighbour = pop_pos[ngh_index]
        cnt_change = np.random.randint(1, col)  # the dim change count
        # randomly determine cnt_change dim to change
        dim_change = np.random.permutation(np.arange(col))
        new_idv = idv
        # diff:the difference between idv and neighbour
        diff = new_idv[dim_change[0:cnt_change]] - neighbour[dim_change[0:cnt_change]]
        # calculate new individual
        new_idv[dim_change[0:cnt_change]] += np.multiply(diff, np.random.rand(cnt_change)) * 2
        bndry_proce(pop_pos[index])
        # new_fitness = objfun.train(new_idv)
        new_fitness = sphere(new_idv)
        evaluation_now += 1
        if new_fitness < fitness[index]:
            pop_pos[index] = new_idv
            fitness[index] = new_fitness
    return evaluation_now


def eva_and_precip(pop_pos, fitness, best_index, evaluation_now):
    # this function completes the evaporation and precipitation operation
    # Args: pop_pos: population position, individual should be a row vector
    # fitness: the fitness value of population
    # best_index: the best individual position index
    # evaluation_now: global control parameter
    row, col = np.shape(pop_pos)
    for index in range(row):
        # evaporation and precipitation with probability p_eva
        if np.random.rand() < p_eva:
            if np.random.rand() < 0.5:
                # move to another position randomly
                pop_pos[index] = np.random.randint(ld, ud, col)
            else:
                pop_pos[index] = pop_pos[best_index]
                cnt_change = np.random.randint(1, col)  # the dim change count
                # randomly determine cnt_change dim to change
                dim_change = np.random.permutation(np.arange(col))
                gaussian = np.random.randn(cnt_change)
                # calculate new individual
                pop_pos[index, dim_change[0:cnt_change]] = np.multiply(pop_pos[index,
                                                                               dim_change[0:cnt_change]], gaussian)
                t = 1
        bndry_proce(pop_pos[index])
        # fitness[index] = objfun.train(pop_pos[index])
        fitness[index] = sphere(pop_pos[index])
        evaluation_now += 1

        return evaluation_now


def bndry_proce(idv):
    # this function is to process the boundary
    # idv: individual of population
    # processing method just for test
    index = np.where(idv > ud)
    idv[index] = ud
    index = np.where(idv < ld)
    idv[index] = ld


def sphere(idv):
    return np.sum(np.multiply(idv, idv))


if __name__ == '__main__':
    hco()
