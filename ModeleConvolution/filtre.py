import numpy as np
from math import sqrt

class Filtre():
    def __init__(self, dim_filtre):
        self.dim_filtre = dim_filtre
        facteur = sqrt(2/(self.dim_filtre[0]*self.dim_filtre[1]*self.dim_filtre[2]))
        self.poids = np.random.randn(self.dim_filtre[0],self.dim_filtre[1], self.dim_filtre[2]) * facteur
        self.biais = 0
    

    def actu_poids_biais(self, poids, biais):
        self.poids = poids
        self.biais = biais



