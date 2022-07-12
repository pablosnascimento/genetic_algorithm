
#from main import f
import random
from matplotlib.pyplot import new_figure_manager
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from src.problem.problem_interface import ProblemInterface

class ClassificationProblem(ProblemInterface):
    def __init__(self, fname):
        # load dataset
        with open(fname, "r") as f:
            lines = f.readlines()

        # For each line l, remove the "\n" at the end using
        # rstrip and then split the string the spaces. After
        # this instruction, lines is a list in which each element
        # is a sublist of strings [s1, s2, s3, ..., sn].
        lines = [l.rstrip().rsplit() for l in lines]

        # Convert the list of list into a numpy matrix of integers.
        lines = np.array(lines).astype(np.int32)

        # Split features (x) and labels (y). The notation x[:, i]
        # returns all values of column i. To learn more about numpy indexing
        # see https://numpy.org/doc/stable/reference/arrays.indexing.html .
        x = lines[:, :-1] #pegando todas colunas exceto a ultima
        y = lines[:, -1] #pegando apenas a ultima coluna

        # Split the data in two sets without intersection.
        self.X_train, self.X_val, self.y_train, self.y_val = \
            train_test_split(x, y, test_size=0.30,
                             stratify=y, random_state=871623)

        # number of features
        self.n_features = self.X_train.shape[1]

        # search space for the values of k and metric
        self.Ks = [1, 3, 5, 7, 9, 11, 13, 15]
        self.metrics = ["euclidean", "hamming", "canberra", "braycurtis"]
        self.name = "classification"
        self.best_solution = []
        self.best_cost = 0.0
        self.worst_cost = 0.0

    def new_individual(self):
        '''
        O indivíduo será híbrido e será composto por padrão binário representando quais features serão consideradas e 
        dois valores categóricos representando o número de vizinhos K e a métrica usada para comparar indivíduos (ver parâmetros).
        '''
        
        individual = [random.randint(0, 1) for i in range(self.n_features)]
        number_K = random.choice(self.Ks)
        metric = random.choice(self.metrics)

        individual.append(number_K)
        individual.append(metric)

        return individual

    def new_population(self, population_size):
        '''
        Criação aleatoria de individuos que preencham a população solicitada
        '''
        population = [self.new_individual() for p in range(population_size)]

        return population

    def fitness(self, individual):
        '''
        Treinar o classificador com os valores do indivíduo e medir a acurácia (número de acertos sobre total de amostras) no conjunto de validação. 
        Soluções para as quais o padrão de bits é completamente 0 (todas as features excluídas) são consideradas inválidas. 
        Neste caso, retorne um valor muito alto para a função objetivo
        '''
        binary_pattern = individual[:self.n_features] #todas caracteristicas
        K = individual[self.n_features:(self.n_features + 1)][0] #penultima posicao do individuo
        metric = individual[(self.n_features + 1):(self.n_features + 2)][0] #ultima posicao do individuo

        # return the indices of the features that are not zero.
        indices = np.nonzero(binary_pattern)[0]

        # check if there is at least one feature available
        if len(indices) == 0:
            return 1e6

        # select a subset of columns given their indices
        x_tr = self.X_train[:, indices]
        x_val = self.X_val[:, indices]

        # build the classifier
        knn = KNeighborsClassifier(n_neighbors=K, metric=metric)
        # train
        knn = knn.fit(x_tr, self.y_train)
        # predict the classes for the validation set
        y_pred = knn.predict(x_val)
        # measure the accuracy
        acc = np.mean(y_pred == self.y_val)

        # since the optimization algorithms minimize,
        # the fitness is defiend as the inverse of the accuracy
        fitness = -acc

        return fitness

    def mutation(self, individual):
        '''
        Inverter um bit aleatório ou escolher um valor aleatório para K dentre os valores
        possíveis ou escolher aleatoriamente uma métrica dentre os valores possíveis
        '''
        mutated = individual[:]
        ind_k = individual[self.n_features]
        ind_metric = individual[self.n_features + 1]

        case = random.uniform(0,1)

        if case < 0.3333:
            #inverter bit aleatorio
            index_bit = random.randint(0, self.n_features-1)
            if mutated[index_bit] == 0:
                mutated[index_bit] = 1
            else:
                mutated[index_bit] = 0
        elif case < 0.6666:
            #escolher valor aleatorio para K, dentre os valores de K possiveis
            mutated[self.n_features] = random.choice([i for i in self.Ks if i != ind_k])
        else:
            #escolher valor aleatorio para metrica
            mutated[self.n_features + 1] = random.choice([i for i in self.metrics if i != ind_metric])

        return mutated

    def crossover(self, p1, p2):
        '''
        No padrão binário, usar crossover de 1 ponto (quem quiser pode usar o crossover de 2 pontos). 
        Nos valores categóricos, faremos um crossover simples: 
        No primeiro filho, o valor de K virá do primeiro pai e o valor da métrica do segundo pai. 
        No segundo filho, faremos o contrário. O valor de K virá do segundo pai e o valor da métrica do primeiro pai
        '''

        #1 ponto de partição que é escolhido aleatoriamente
        index = random.randint(1, self.n_features - 2)

        #do inicio ate o indice
        beginP1 = p1[:index]
        beginP2 = p2[:index]

        #do indice ate o final
        endP1 = p1[index:]
        endP2 = p2[index:]

        c1 = beginP1 + endP2
        c2 = beginP2 + endP1

        return c1, c2

	#Nessa classificador essa função não precisa, não faz sentido
    def plot(self, individual):

        print("Solução: ",  individual)

        pass
