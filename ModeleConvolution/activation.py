from abc import ABC, abstractmethod
import numpy as np


class activation(ABC):
    @staticmethod
    @abstractmethod
    def calcul(valeur):
        pass

    @staticmethod
    @abstractmethod
    def derive(valeur):
        pass

    @staticmethod
    @abstractmethod
    def init_poids(dim):
        pass


class ReLU(activation):
    @staticmethod
    def calcul(valeur):
        return np.maximum(0, valeur)

    @staticmethod
    def derive(valeur):
        return (valeur > 0).astype(valeur.dtype)

    @staticmethod
    def init_poids(dim):
        # Initialisation He : adaptee a ReLU
        if len(dim) == 2:
            n_in = dim[0]
        else:
            n_in = int(np.prod(dim[1:])) if len(dim) >= 3 else int(np.prod(dim))
        facteur = np.sqrt(2.0 / n_in)
        return (np.random.randn(*dim) * facteur).astype(np.float32)


class Sigmoide(activation):
    @staticmethod
    def calcul(valeur):
        return 1.0 / (1.0 + np.exp(-valeur))

    @staticmethod
    def derive(valeur):
        sig = Sigmoide.calcul(valeur)
        return sig * (1.0 - sig)

    @staticmethod
    def init_poids(dim):
        # Initialisation Xavier/Glorot
        if len(dim) == 2:
            n_in, n_out = dim[0], dim[1]
        else:
            n_in = int(np.prod(dim[1:]))
            n_out = dim[0]
        variance = np.sqrt(2.0 / (n_in + n_out))
        return (np.random.randn(*dim) * variance).astype(np.float32)


class Tanh(activation):
    @staticmethod
    def calcul(valeur):
        return np.tanh(valeur)

    @staticmethod
    def derive(valeur):
        return 1.0 - np.tanh(valeur) ** 2

    @staticmethod
    def init_poids(dim):
        if len(dim) == 2:
            n_in, n_out = dim[0], dim[1]
        else:
            n_in = int(np.prod(dim[1:]))
            n_out = dim[0]
        variance = np.sqrt(2.0 / (n_in + n_out))
        return (np.random.randn(*dim) * variance).astype(np.float32)


class Softmax(activation):
    """
    ATTENTION : la derive renvoie 1 car on couple toujours Softmax
    avec la cross-entropy. Le gradient combine softmax + cross-entropy
    vaut simplement (y_pred - y_true), injecte directement dans Modele.backward.
    Ne PAS utiliser cette classe ailleurs qu'en derniere couche.
    """

    @staticmethod
    def calcul(valeur):
        # Stable : on soustrait le max sur l'axe des classes
        maximum = np.max(valeur, axis=-1, keepdims=True)
        exps = np.exp(valeur - maximum)
        diviseur = np.sum(exps, axis=-1, keepdims=True)
        return exps / diviseur

    @staticmethod
    def derive(valeur):
        # Pass-through : le gradient softmax+CE est deja calcule en amont
        return np.ones_like(valeur)

    @staticmethod
    def init_poids(dim):
        if len(dim) == 2:
            n_in, n_out = dim[0], dim[1]
        else:
            n_in = int(np.prod(dim[1:]))
            n_out = dim[0]
        variance = np.sqrt(2.0 / (n_in + n_out))
        return (np.random.randn(*dim) * variance).astype(np.float32)
