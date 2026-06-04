from abc import ABC, abstractmethod
from math import exp
class activation(ABC):
    @abstractmethod
    @staticmethod
    def calcul(valeur):
        pass
    
    @staticmethod
    @abstractmethod
    def derive(valeur):
        pass

class ReLU(activation):

    @staticmethod
    def calcul(valeur):
        return max(0,valeur)
    
    @staticmethod
    def derive(valeur):
        if valeur >= 0:
            return 1
        else:
            return 0

class Sigmoide(activation):
    @staticmethod
    def calcul(valeur):
        return 1/(1 + exp(-valeur))
    
    @staticmethod
    def derive(valeur):
        return Sigmoide.calcul(valeur)*(1-Sigmoide.calcul(valeur))

class Tanh(activation):
    @staticmethod
    def calcul(valeur):
        return (1 - exp(-2*valeur))/(1 + exp(-2*valeur))
    
    @staticmethod
    def derive(valeur):
        return 1 - Tanh.calcul(valeur)**2

class LeakyRelu(activation):
    def __init__(self, fuite):
        self.fuite = fuite

    def calcul(self, valeur):
        if valeur < 0:
            return self.fuite * valeur
        else:
            return valeur
    
    def derive(self, valeur):
        if valeur < 0:
            return self.fuite
        else:
            return 1

class Softmax(activation):
    @staticmethod
    def calcul(valeur):
        return (1 - exp(-2*valeur))/(1 + exp(-2*valeur))
    
    @staticmethod
    def derive(valeur):
        return 1 - Tanh.calcul(valeur)**2
