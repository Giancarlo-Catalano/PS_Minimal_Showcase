import numpy as np

import utils
from PRef import PRef

def get_normalised_fitness_array(fitness_array: np.ndarray) -> np.ndarray:
    normalised = utils.remap_array_in_zero_one(fitness_array)
    return normalised / np.sum(normalised)

def get_normalised_pRef(pRef: PRef) -> PRef:
    return PRef.from_just_matrix(solution_matrix=pRef.full_solution_matrix,
                                 fitness_array=get_normalised_fitness_array(pRef.fitness_array),
                                 search_space=pRef.search_space)

def p_a(var_A: int, val_A: int, context_pRef: PRef) -> float:
    return float(np.sum(context_pRef.fitness_array,
                        where=context_pRef.full_solution_matrix[:, var_A] == val_A))

def p_a_and_b(var_A: int, val_A: int,
              var_B: int, val_B: int,
              context_pRef: PRef) -> float:
    where = ((context_pRef.full_solution_matrix[:, var_A] == val_A) &
             (context_pRef.full_solution_matrix[:, var_B] == val_B))
    return float(np.sum(context_pRef.fitness_array, where=where))


def get_conditional_pRef(var_A: int, val_A: int, context_pRef: PRef) -> PRef:
    where = context_pRef.full_solution_matrix[:, var_A] == val_A
    solution_matrix = context_pRef.full_solution_matrix[where]
    fitnesses = get_normalised_fitness_array(context_pRef.fitness_array[where])
    return PRef.from_just_matrix(solution_matrix,
                                 fitness_array=fitnesses,
                                 search_space=context_pRef.search_space)


def get_uc_linkage_table(original_pRef: PRef) -> np.ndarray:
    amount_of_vars = original_pRef.search_space.amount_of_parameters
    cardinalities = original_pRef.search_space.cardinalities
    levels = [list(range(cardinality)) for cardinality in cardinalities]

    global_pRef = get_normalised_pRef(original_pRef)  # note that this NEEDS to be normalised!
    marginal_probabilities = [np.array([p_a(var, val, global_pRef)
                                        for val in levels_for_var])
                              for var, levels_for_var in enumerate(levels)]

    marginal_entropies = [-np.sum(marginals * np.log(marginals)) for marginals in marginal_probabilities]

    conditional_pRefs = [[get_conditional_pRef(var, val, global_pRef)
                          for val in levels_for_var]
                         for var, levels_for_var in enumerate(levels)]

    def p_aIb(var_A, val_A, var_B, val_B):
        given_b = conditional_pRefs[var_B][val_B]
        return p_a(var_A, val_A, given_b)

    def get_H_AIB(var_A, var_B):
        def term(val_A, val_B):
            p_a_comma_b = p_a_and_b(var_A, val_A, var_B, val_B, global_pRef)
            p_a_bar_b = p_aIb(var_A, val_A, var_B, val_B)
            return p_a_comma_b * np.log(p_a_bar_b)
        return -np.sum([term(val_A, val_B)
                        for val_A in levels[var_A]
                        for val_B in levels[var_B]])

    def get_symmetric_uncertainty_coefficient(var_A, var_B) -> float:
        H_A = marginal_entropies[var_A]
        H_B = marginal_entropies[var_B]
        H_AIB = get_H_AIB(var_A, var_B)
        H_BIA = get_H_AIB(var_B, var_A)
        return 1-(H_AIB + H_BIA)/(H_A + H_B)


    linkage_table = np.zeros((amount_of_vars, amount_of_vars), dtype=float)
    for var_X in range(amount_of_vars):
        for var_Y in range(var_X+1, amount_of_vars):
            linkage_table[var_X][var_Y] = get_symmetric_uncertainty_coefficient(var_X, var_Y)

    linkage_table += linkage_table.T
    #np.fill_diagonal(linkage_table, 1.0)
    return linkage_table


