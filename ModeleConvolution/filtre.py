import numpy as np
from math import sqrt

class Filtre():
    def __init__(self, dim_filtre, activation):
        print(dim_filtre)
        self.dim_filtre = dim_filtre
        self.poids = activation.init_poids(dim_filtre)
        self.biais = 0
    

    def actu_poids_biais(self, poids, biais):
        self.poids = poids
        self.biais = biais



