import math
from numpy.core.records import fromrecords
from src.problem.function_metrics import Metrics
import random
import numpy as np
import time
import matplotlib.pyplot as plt

def genetic_algorithm(problem, population_size, n_generations, mutation_rate=0.1):

    plt.ion()

    time_start = time.perf_counter()

    population = problem.new_population(population_size)
    pop_fitness = problem.fitness_population(population)
    new_population = []
    
    best_pop_solution = []
    best_pop_fit = 99999999

    best_solution = []
    best_fit = 99999999
    worst_fit = 0   

    if(problem.name == "classification"):
        worst_fit = -99999999
    
    generation = 0
    generations_without_change_best = 0
    m_rate_aux = mutation_rate

    #cria n gerações até o limite estabelecido
    while generation < n_generations:

        generation += 1

        for idx in range(0, population_size, 2):
            
            x, fitx, y, fity = problem.selection(population, pop_fitness)

            child1, child2 = problem.crossover(x, y)

            #se probabilidade de 20% (mutation_rate = 0.2) for obtida, faz mutação
            if random.uniform(0,1) < mutation_rate:
                child1 = problem.mutation(child1)
                child2 = problem.mutation(child2)

            new_population.append(child1)
            new_population.append(child2)
        
        #o melhor individuo da população anterior deve permanecer na nova população (Elitismo)
        best_pop_solution, best_pop_fit = problem.selection_best_fitness(population, pop_fitness)

        pop_fitness = problem.fitness_population(new_population)
        
        worst_tour, worst_pop_fit = problem.selection_worst_fitness(new_population, pop_fitness)

        #insere o melhor da população atual na nova geração
        index = pop_fitness.index(worst_pop_fit)
        new_population[index] = best_pop_solution
        pop_fitness[index] = best_pop_fit
        
        #garante que o melhor tour já encontrado não se perca
        if(best_pop_fit < best_fit):
            best_solution = best_pop_solution
            best_fit = best_pop_fit
            generations_without_change_best = 0
            
            #volta a probabilidade de mutacao para o padrao
            mutation_rate = m_rate_aux
        else:
            generations_without_change_best += 1
        
        #a cada 15 iterações sem melhora aumento a probabilidade de mutação até 0.6
        if generations_without_change_best != 0 and generations_without_change_best % 15 == 0 and mutation_rate < 0.6:
            mutation_rate += 0.1

        #guardar o pior fitness encontrado
        if(worst_fit < worst_pop_fit):
            worst_fit = worst_pop_fit

        population = new_population
        new_population = []
        
        if problem.name == "classification":
            print('Geração', generation, ' - best fit/solution:', (-1)*best_fit, '/', best_solution, "Time:", time.perf_counter() - time_start)
        else:
            plt.pause(0.0001)
            plt.clf()
            problem.plot(best_solution)
            print('Geração', generation, ' - best fit/solution:', best_fit, '/', best_solution, "Time:", time.perf_counter() - time_start)
        
        

    time_end = time.perf_counter()

    #como na classificacao o valor de fitness é negativado para maximização do genético, inverter o sinal e o resultado
    if problem.name == "classification":
        aux = best_fit
        best_fit = (-1)*best_fit
        worst_fit = (-1)*worst_fit
        best_fit = worst_fit
        worst_fit = (-1)*aux

    problem.best_solution.append(best_solution)
    problem.best_cost = best_fit
    problem.worst_cost = worst_fit

    function_metrics = Metrics(problem.name, worst_fit, best_fit, 0.0, 0.0, (time_end - time_start))

    print('best fitness:', best_fit)
    print('best tour:', best_solution)

    return best_solution, function_metrics
