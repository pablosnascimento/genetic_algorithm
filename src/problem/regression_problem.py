
from scipy.sparse.construct import rand
from src.problem.problem_interface import ProblemInterface

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import numpy as np
import math
import matplotlib.pyplot as plt
import random

class RegressionProblem(ProblemInterface):
    def __init__(self, fname):
        '''
        Lê o arquivo com os pontos e inicializa a estrutura inicial.
        '''

        with open(fname, "r") as f:
            points = f.readlines()

        points = [l.rstrip().rsplit() for l in points]
        points = np.array(points).astype(np.float)
        size = len(points)

        self.best_solution = []
        self.best_cost = 0.0
        self.worst_cost = 0.0

        self.individual_size = size
        self.neutral_individual = [x for x in range(size)]
        self.dataset = points
        self.name = "regression"

        pass

    def function_xi(self, individual, x, n_components=4):
        '''
        '''
        a0 = individual[0]
        a1 = individual[1]
        b1 = individual[2]
        a2 = individual[3]
        b2 = individual[4]
        a3 = individual[5]
        b3 = individual[6]
        a4 = individual[7]
        b4 = individual[8]
        
        s1 = a1 * (math.sin(1*x)) + b1 * (math.cos(1*x))
        s2 = a2 * (math.sin(2*x)) + b2 * (math.cos(2*x))
        s3 = a3 * (math.sin(3*x)) + b3 * (math.cos(3*x))
        s4 = a4 * (math.sin(4*x)) + b4 * (math.cos(4*x))
        
        f = a0 + s1 + s2 + s3 + s4
        
        return f

    def fitness(self, individual):
        '''
        A função objetivo será dada pelo erro médio quadrático. Formula matematica na descrição do problema
        Deve calcular o valor total do erro para o individuo fornecido.
        '''

        erro = []
        n = len(self.dataset)
        
        for index in range(n):
            
            xi = self.dataset[index][0]
            yi = self.dataset[index][1]

            f_xi = self.function_xi(individual, xi)
            
            erro.append(math.pow(yi - f_xi, 2))

        erroMedio = np.mean(erro)
        
        return erroMedio

    def fit(self, individual, points_to_plot):
        '''
        Função que aplica a regressão pela função proposta a cada x e retorna o y correspondente.
        '''
        n = len(points_to_plot)
        y = []

        for index in range(n):
            
            xi = points_to_plot[index]
            
            y.append(self.function_xi(individual, xi))
        
        return y

    def new_individual(self):
        '''
        Criação aleatória de um individuo.
        Array com 9 valores reais representando os coeficientes 𝒂𝟎, e 𝒂𝒊 e 𝒃𝒊 para os 4 componentes
        Os valores dos coeficientes estará no intervalo [-100;100]
        '''
        
        individual = [random.uniform(-100, 100) for i in range(9)]

        return np.array(individual)
    
    def new_population(self, population_size):
        '''
        Criação aleatoria de individuos que preencham a população solicitada
        '''
        population = [self.new_individual() for p in range(population_size)]

        return population

    def mutation(self, individual):
        '''
        Em 50% dos casos, atualize o valor de um coeficiente para um valor aleatório no conjunto de valores possíveis. 
        Nos outros casos, some ao valor de um coeficiente um valor aleatório amostrado de uma distribuição normal padrão (média 0 e desvio padrão 1)
        '''

        #definindo aleatoriamente qual coeficiente será alterado.
        indexCoef = random.randint(0, 8)
        
        #copia do individuo para nao interferir na populacao original
        ind_copy = individual[:]

        caso = random.uniform(0, 1)

        if caso < 0.5:
            #Atualizar um dos coeficientes para um valor aleatorio
            coeficienteAleatorio = random.uniform(-100,100)

            ind_copy[indexCoef] = coeficienteAleatorio
        else:
            #Somar ao valor de m coeficiente um valor aleatório amostrado de uma distribuição normal padrão (média 0 e desvio padrão 1)
            number = np.random.normal(0, 1)
            final = ind_copy[indexCoef] + number

            #garantir que a soma não ultrapasse os limites definidos de 100.
            if (final < -100):
                final = -100
            elif (final > 100):
                final = 100

            ind_copy[indexCoef] = final

        return ind_copy

    def crossover(self, p1, p2):
        '''
        Sejam p1 e p2 dois pontos escolhidos para produzirem filhos. (Precisam ser np.array!)
        A operação de crossover será dada pela média ponderada c1 = p1 α + p2 (1 - α) e c2 = p2 α + p1 (1 - α), onde α é um número aleatório amostrado de uma distribuição uniforme entre 0 e 1
        '''

        #α é um número aleatório amostrado de uma distribuição uniforme entre 0 e 1
        alfa = random.uniform(0, 1)
        
        c1 = p1 * alfa + p2 * (1 - alfa)
        c2 = p2 * alfa + p1 * (1 - alfa)

        return c1, c2

    def plot(self, individual):
        '''
        Impressao do grafico com os pontos e a função de regressão aplicada.
        '''

        x = self.dataset[:, 0] #pegando apenas a primeira coluna (pontos x)
        y = self.dataset[:, 1] #pegando apenas a segunda coluna (pontos y)

        # generate points used to plot
        x_plot = np.linspace(np.min(x), np.max(x), 50)

        #coeficientes otimizados com os valores de x
        fx = self.fit(individual, x_plot)

        plt.plot(x_plot, fx, color='cornflowerblue')
        plt.scatter(x, y, color='red', s=10)

        #plt.legend(loc='lower left')

        plt.show()