from abc import ABC, abstractmethod
from math import exp,sqrt,prod
import numpy as np

class activation(ABC):
    @abstractmethod
    @staticmethod
    def calcul(valeur):
        pass
    
    @abstractmethod
    @staticmethod
    def derive(valeur):
        pass
    
    @abstractmethod
    @staticmethod
    def init_poids(dim):
        pass
    
class ReLU(activation):

    @staticmethod
    def calcul(valeur):
        return np.maximum(0,valeur)
    
    @staticmethod
    def derive(valeur):
        return 1.0 * (valeur > 0)
    
    @staticmethod
    def init_poids(dim):
        if len(dim) == 2:
            n_in = dim[0]
        elif len(dim) == 3:
            n_in = np.prod(dim)
        facteur = np.sqrt(2.0 / n_in)
        return np.random.randn(*dim) * facteur

class Sigmoide(activation):
    @staticmethod
    def calcul(valeur):
        return 1/(1 + np.exp(-valeur))
    
    @staticmethod
    def derive(valeur):
        sig = Sigmoide.calcul(valeur)
        return sig * (1.0 - sig)
    
    @staticmethod
    def init_poids(dim):
        variance_xavier = np.sqrt(2.0 / (sum(dim)))
        return np.random.randn(*dim) * variance_xavier 

class Tanh(activation):
    @staticmethod
    def calcul(valeur):
        return np.tanh(valeur)
    
    @staticmethod
    def derive(valeur):
        return 1 - Tanh.calcul(valeur)**2
    
    @staticmethod
    def init_poids(dim):
        variance_xavier = np.sqrt(2.0 / (sum(dim)))
        return np.random.randn(*dim) * variance_xavier 

class LeakyRelu(activation):
    def __init__(self, fuite):
        self.fuite = fuite

    def calcul(self, valeur):
        return np.maximum(self.fuite * valeur, valeur)
    
    def derive(self, valeur):
        return 1.0 * (valeur > 0) + self.fuite * (valeur <= 0)
    
    @staticmethod
    def init_poids(dim):
        if len(dim) == 2:
            n_in = dim[0]
        elif len(dim) == 3:
            n_in = np.prod(dim)
        facteur = np.sqrt(2.0 / n_in)
        return np.random.randn(*dim) * facteur
    
    
class Softmax(activation):
    @staticmethod
    def calcul(valeur):
        diviseur = np.sum(np.exp(valeur))
        return np.exp(valeur)/diviseur
    
    @staticmethod
    def init_poids(dim):
        variance_xavier = np.sqrt(2.0 / (sum(dim)))
        return np.random.randn(*dim) * variance_xavier 
    
    @staticmethod
    def derive(valeur):
        pass