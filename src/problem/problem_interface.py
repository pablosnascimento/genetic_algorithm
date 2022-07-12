
from abc import ABC, abstractmethod
import random
import numpy as np
from numpy.core.numeric import array_equal

# ProblemInterface is an abstract base class (ABC). The classes that
# inherit from ProblemInterface need to implement the abstract methods
# in order to be instantiable.
class ProblemInterface(ABC):
    #Retorna um valor de fitness baseado no indivudio informado
    @abstractmethod
    def fitness(self, individual):
        pass

	#Retorna um novo individuo criado. É aqui que busa o elemento aleatorio???
    @abstractmethod
    def new_individual(self):
        pass

    #Retorna um novo individuo criado. É aqui que busa o elemento aleatorio???
    @abstractmethod
    def new_population(self, population_size):
        pass

	#Retorna um novo individuo alterado pela função de mutacao
    @abstractmethod
    def mutation(self, individual):
        pass

	#Retorna dois filhos resultantes do crossover d1 p1 e p2
    @abstractmethod
    def crossover(self, p1, p2):
        pass

    def fitness_population(self, population):
        '''
        Calcular o fitness de todos elementos da população
        '''
        pop_fitness = []
        for i in range(len(population)):
            fit = self.fitness(population[i])
            pop_fitness.append(fit)
        
        return pop_fitness
    
    def selection_best_fitness(self, population, pop_fitness):
        '''
        Retorna o melhor valor de fitness da população
        '''
        min_value = min(pop_fitness)
        min_index = pop_fitness.index(min_value)

        return population[min_index], min_value

    def selection_worst_fitness(self, population, pop_fitness):
        '''
        Retorna o pior valor de fitness da população. Util para ser trocado
        '''
        max_value = max(pop_fitness)
        max_index = pop_fitness.index(max_value)

        return population[max_index], max_value

    def selection(self, population, pop_fitness):
        '''
        Seleciona os dois elementos por torneio. 
        '''
        x = []
        y = []
        x2 = []
        y2 = []

        #tendo o melhor fitness garantimos que a cada iteração o melhor fitness é considerado na geração de filhos tendendo a uma boa prole rs.
        x, fitx = self.selection_best_fitness(population, pop_fitness)
        index_x = pop_fitness.index(fitx)

        pai1 = x
        fit1 = fitx

        #busca um novo item diferente aleatoriamente, mas tenta verificar 10 vezes se forem iguais buscando outro. Senao, retorna ele mesmo
        count = 0
        fita = random.choice(pop_fitness)
        while (fitx == fita) and count < 10:
            fita = random.choice(pop_fitness)
            count += 1
        else:
            index_a = pop_fitness.index(fita)

        #busca um novo item diferente aleatoriamente, mas tenta verificar 10 vezes se forem iguais buscando outro. Senao, retorna ele mesmo
        count = 0
        fitb = random.choice(pop_fitness)
        while (fitx == fitb or fita == fitb) and count < 10:
            fitb = random.choice(pop_fitness)
            count += 1
        else:
            index_b = pop_fitness.index(fitb)
        
        pai2, fit2 = self.tournament(fita, index_a, fitb, index_b, population)

        return pai1, fit1, pai2, fit2
    
    def tournament(self, fitx, index_x, fity, index_y, population):
        '''
        Torneio entre dois elementos: avalia o fitness de ambos e retorna o par (best_solution, value)
        '''

        if fitx < fity:
            return population[index_x], fitx
        else:
            return population[index_y], fity