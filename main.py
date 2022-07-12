
import argparse
from src.problem.regression_problem import RegressionProblem
from src.problem.classification_problem import ClassificationProblem
from src.problem.tsp_problem import TSPProblem
from src.algorithm.genetic_algorithm import genetic_algorithm
import numpy as np
import time
import matplotlib.pyplot as plt

from src.problem.function_metrics import Metrics
from sklearn.pipeline import make_pipeline
from multiprocessing import Pool

def generate_report(array_metrics, problemName): 
    '''
    Imprime as estatisticas das execuções
    '''
    result = []
    max = []
    min = []
    media = []
    std = []
    time = []

    f = open("results.txt", "a")
    header = f"\t\t\tMax \t\t|\tMin \t\t|\tMean \t\t|\tDeviation \t|\tTime\t"
    print(header)
    f.write(header)
    f.write("\n")

    name = array_metrics[0].name
    for i in range(len(array_metrics)):
        
        item = array_metrics[i]
        
        if name == item.name:
            if problemName == 'classification':
                result.append(item.max)
            else:
                result.append(item.min)
            
            time.append(item.tempo_total)
        
        if name != item.name or i == len(array_metrics) - 1:
            media = np.mean(result)
            std = np.std(result)
            max = np.max(result)
            min = np.min(result)
            time = np.mean(time)
            
            msg = f"\t{name}:\t%.15s\t|\t%.15s\t|\t%.15s\t|\t%.15s\t|\t%.15s" % (max, min, media, std, time)
            print(msg)
            f.write(msg)
            f.write("\n")

            max = []
            min = []
            media = []
            std = []
            time = []

            if problemName == 'classification':
                result.append(item.max)
            else:
                result.append(item.min)
            
            time.append(item.tempo_total)

            name = item.name

    f.close()
    pass

def build_problem(problem_name):
    if problem_name == "classification":
        return ClassificationProblem("data/german_statlog/german.data-numeric")
    elif problem_name == "regression":
        return RegressionProblem("data/regression/data-3.txt")
    elif problem_name == "tsp":
        return TSPProblem("data/tsp/tsp-30.txt")
    else:
        raise NotImplementedError()


def read_command_line_args():
    parser = argparse.ArgumentParser(
        description='Optimization with genetic algorithms.')

    parser.add_argument('-p', '--problem', default='classification',
                        choices=["classification", "regression", "tsp", "all"])
    parser.add_argument('-n', '--n_generations', type=int,
                        default=1000, help='number of generations.')
    parser.add_argument('-s', '--population_size', type=int,
                        default=200, help='population size.')
    parser.add_argument('-m', '--mutation_rate', type=float,
                        default=0.2, help='mutation rate.')

    args = parser.parse_args()
    return args

def main():

    plt.ion()

    args = read_command_line_args()

    problems = []
    if args.problem == "all":
        problems = ["tsp", "regression", "classification"]
    else:
        problems.append(args.problem)
    
    array_metrics = []

    for p in range(len(problems)):
    
        problem = build_problem(problems[p])

        size = args.population_size
        generations = args.n_generations
        m_rate = args.mutation_rate

        if problem.name == "classification":
            size = 100 #200
            generations = 250 #500

        for i in range(5):
            print("__ " + problem.name + " Iteração:", i+1, "__")

            best_solution, genetic_metrics = genetic_algorithm(
            problem,
            population_size=size,
            n_generations=generations,
            mutation_rate=m_rate)

            #grafico interativo
            if(problem.name != "classification"):
                plt.pause(0.0001)
                plt.clf()
                problem.plot(best_solution)
                file = problem.name + "_solucao" + str(i+1) + ".png"
                plt.savefig(file)

            array_metrics.append(genetic_metrics)
        
    #receber a saidas das 5 rodadas e gerar os relatorios de uma vez
    generate_report(array_metrics, args.problem)

    print("OK!")


if __name__ == "__main__":
    main()
