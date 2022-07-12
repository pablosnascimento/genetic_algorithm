
from os import pathsep

from numpy.lib.function_base import copy
from scipy.sparse.construct import rand
from src.problem.function_metrics import Metrics
import numpy as np
from src.problem.problem_interface import ProblemInterface
from scipy.spatial import distance
import matplotlib.pyplot as plt
import seaborn as sns
import random
import copy

class TSPProblem(ProblemInterface):
    def __init__(self, fname):
        '''
        Lê o arquivo com as cidades e inicializa a estrutura inicial
        #representação do vetor: [0 1 4 2 0] visitando as cidades 0 -> 1 -> 4 -> 2 -> 0. Logo, todas as cidades mais 1
        '''
        with open(fname, "r") as f:
            cidades = f.readlines()

        cidades = [l.rstrip().rsplit() for l in cidades]
        cidades = np.array(cidades).astype(np.float)
        size = len(cidades)
        
        self.best_solution = []
        self.best_cost = 0.0
        self.worst_cost = 0.0
        self.individual_size = size
        self.neutral_individual = [x for x in range(size)]
        #print('ind neutro = ', self.neutral_individual)
        self.dataset = cidades
        self.name = "tsp"

        pass

    #No TSP a fitness function é medida pela distância após percorridas todas cidades. Ou seja, a soma das distâncias entre elas (não esquecer de voltar ao inicio)
    #individual: vetor de inteiros representando a ordem das cidades [0 1 4 2 5 3 0]
    #tuple: verificar se depois nao seria melhor ja nascer como tuple ao inves de converter toda vez
    def fitness(self, individual):
        '''
        Deve calcular o valor total do percurso para o individuo fornecido.
        '''
        firstCity = self.getCity(individual[0]) 
        dist = 0.0
        index = 0
        for index, cit in enumerate(individual):
            begin = tuple(self.getCity(cit))

            if (index + 1 < len(individual)):
                end = tuple(self.getCity(individual[index + 1])) 
            else:
                end = firstCity

            #soma as distancias entre todas as cidades
            dist = dist + distance.euclidean(begin, end)
        return dist

    def fitness_population(self, population):
        '''
        Calcular o fitness de todos elementos da população
        '''
        pop_fitness = []
        for i in range(len(population)):
            fit = self.fitness(population[i])
            pop_fitness.append(fit)
        
        return pop_fitness

    def new_individual(self):
        '''
        Criação aleatória de um individuo
        '''

        #nao precisa variar o primeiro, pois a primeira cidade pode ser tomada como sendo sempre a mesma
        individual = [0]
        individual.extend(random.sample([i for i in range(self.individual_size) if i != 0], self.individual_size - 1))

        return individual

    def new_population(self, population_size):
        '''
        Criação aleatoria de individuos que preencham a população solicitada
        '''
        population = [self.new_individual() for p in range(population_size)]

        return population

    #Trocar duas cidades aleatorias de ordem
    def mutation(self, individual):
        '''
        Altera o individuo e retorna sua mutação
        Probabilidade de mutação sugerida entre 10% e 20%
        '''
        #Se o numero de trocas no vetor fosse probabilistico...
        #n = len(self.neutral_individual)
        
        # probability = 0.4
        # #se a entrada não estiver correta como decimal
        # while probability > 1:
        #     probability = probability / 10

        # n_changes = int(probability * n)

        # #o numero de elemtnso tem que ser par pra poder serem trocados entre si
        # if (n_changes % 2 == 1):
        #     n_changes = n_changes - 1
        
        # #troca no minimo 2 elementos
        # if (n_changes == 1):
        #     n_changes = 2


        #como o numero de trocas no vetor é 2, fica fixo
        n_changes = 2 

        #Exceto o primeiro elemento 0 representando a primeira cidade
        change = random.sample(self.neutral_individual[1:], n_changes)

        # #a cada passada troca o par de posição, por isso o passo é 2 em 2
        for idx in range(0, n_changes, 2):
            first = individual[change[idx]]
            second = individual[change[idx+1]]
        
            #print("antes da troca", individual)
            individual[change[idx]] = second
            individual[change[idx+1]] = first
            #print("apos a troca", individual)

        return individual

    #Order 1 Crossover
    def crossover(self, pai1, pai2):
        '''
        Efetuar operação de crossover com os dois elementos e gerar dois novos.
        '''
        #TSP nao pode repetir, entao tem que validar antes de adicionar, só trocar nao funciona!
        #1 ponto de partição que é escolhido aleatoriamente

        #Importante: não precisa mudar a primeira posição 0, pois vamos considerar que partiremos sempre da cidade 0. 
        #A criação de um novo individuo aleatorio ja trata isso tambem.
        p1 = pai1[1:]
        p2 = pai2[1:]

        index_begin = random.randint(1, len(p1)-1)
        index_end = random.randint(1, len(p1)-1)
        
        while (index_begin == index_end):
            index_end = random.randint(0, len(p1)-1)

        #troca se necessario
        if index_begin > index_end:
            aux = index_begin
            index_begin = index_end
            index_end = aux

        # #do inicio ate o indice 1
        beginP1 = p1[:index_begin]
        beginP2 = p2[:index_begin]

        #do indice 2 ate o final
        endP1 = p1[index_end:]
        endP2 = p2[index_end:]

        #configuracao boa: begin exato e end pega um indice antes. indece 9 posiciona no 8 o que faz sempre gerar um cross com pelo menos uma casa no final
        middle1 = p1[index_begin:index_end]
        middle2 = p2[index_begin:index_end]

        #preencheno o filho 1
        #auxiliares começam como final, conforme sequencia sera percorrida no for do cross
        p1_aux = copy.deepcopy(endP1)
        p1_aux.extend(beginP1)
        p1_aux.extend(middle1)

        p2_aux = copy.deepcopy(endP2)
        p2_aux.extend(beginP2)
        p2_aux.extend(middle2)

        #completa o middle ate o final, depois corta a terceira parte e cola no inicio |--|XX|YY| -> |YY|--|XX|
        for idx in range(len(p2_aux)):
            i = p2_aux[idx]
            
            if not(i in middle1):
                middle1.append(i)
            
            i = p1_aux[idx]
            if not(i in middle2):
                middle2.append(i)

        #adicionamos novamente a primeira cidade removida no inicio do crossover
        middle1 = [0] + middle1[index_end:] + middle1[:index_end]
        middle2 = [0] + middle2[index_end:] + middle2[:index_end]

        return middle1, middle2

    def plot(self, individual):
        '''
        fonte: https://gist.github.com/payoung/6087046
        path: List of lists with the different orders in which the nodes are visited
        points: coordinates for the different nodes
        num_iters: number of paths that are in the path list
        
        '''
        sns.set()
        paths = individual #o individual é o caminho escolhido

        # Pontos iniciais que representam as cidades e plota no grafico
        x = []
        y = []
        
        x = self.dataset[:, 0] #pegando apenas a primeira coluna (pontos x)
        y = self.dataset[:, 1] #pegando apenas a segunda coluna (pontos y)

        plt.plot(x, y, '.r')

        # Set a scale for the arrow heads (there should be a reasonable default for this, WTF?)
        a_scale = float(max(x))/float(100)

        # Draw the older paths, if provided
        num_iters = 1
        '''
        if num_iters > 1:

            for i in range(1, num_iters):
        '''
        # Transform the old paths into a list of coordinates
        xi = [] 
        yi = []

        for j in paths:
            xi.append(self.dataset[j][0])
            yi.append(self.dataset[j][1])

        plt.arrow(xi[-1], yi[-1], (xi[0] - xi[-1]), (yi[0] - yi[-1]), 
                head_width = a_scale, color = 'cornflowerblue', 
                length_includes_head = True, 
                width = 0.001/float(num_iters))
        for i in range(0, len(x) - 1):
            plt.arrow(xi[i], yi[i], (xi[i+1] - xi[i]), (yi[i+1] - yi[i]),
                    head_width = a_scale, color = 'cornflowerblue', length_includes_head = True,
                    width = 0.001/float(num_iters))
        
        plt.show()

        pass

    def printAllCities(self):
        for p in self.dataset:
            print(p)
    
    def getCity(self, number):
        city = self.dataset[number]
        return city