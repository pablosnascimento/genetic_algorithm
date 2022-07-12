
import numpy as np

class Metrics:
    def __init__(self, name, max, min, media, desvio_padrao, tempo_total):
        self.name = name
        self.max = max
        self.min = min
        self.media = media
        self.desvio_padrao = desvio_padrao
        self.tempo_total = tempo_total

    def print(self):
        print(" - Metricas - ")
        print("Max:", self.max)
        print("Min:", self.min)
        print("Media:", self.media)
        print("Desvio Padrão:", self.desvio_padrao)
        print("Tempo total:", self.tempo_total)
    
    def printTable(self):
        print(" - Metricas parciais - ")
        print("\tMax\t|\tMin\t|\tMean\t|\tDeviation\t|\tTime\t")
        print(f"\t{self.max}\t|\t{self.min}\t|\t{self.media}\t|\t{self.desvio_padrao}\t|\t{self.tempo_total}")