from abc import ABC, abstractmethod
from filtre import *

class Couche(ABC):
    @abstractmethod
    def forward(self, entree):
        pass

    @abstractmethod
    def backward(self, gradient, learning_rate):
        pass

class Couche_convolution(Couche):
    def __init__(self, nb_filtre, dim_filtre):
        assert dim_filtre[1] % 2 == 1 and dim_filtre[2] % 2 == 1, "Les dimensions du filtres posent probleme"
        self.tab_filtre = []
        self.dim_filtre = dim_filtre
        self.nb_filtre = nb_filtre
        for _ in range(nb_filtre):
            self.tab_filtre.append(Filtre(dim_filtre))
        self.padding_hauteur = (dim_filtre[1] - 1) // 2
        self.padding_cote = (dim_filtre[2] - 1) // 2
        self.entree = None
    
    def forward(self, entree):
        assert entree.shape[0] == self.dim_filtre[0], "Probleme la taille du filtre ne corresspond pas avec la taille de l'entrée"
        self.entree = entree
        configuration_padding = ((0,0), (self.padding_hauteur,self.padding_hauteur), (self.padding_cote, self.padding_cote))
        mat_image = np.zeros((self.nb_filtre, entree.shape[1], entree.shape[2]))
        entree_avec_padding = np.pad(entree, pad_width=configuration_padding, mode="constant", constant_values=0)
        dim_entree_avec_padding = entree_avec_padding.shape
        for ind_filtre in range(self.nb_filtre):
            filtre = self.tab_filtre[ind_filtre]
            for ind_ligne in range(self.padding_hauteur, dim_entree_avec_padding[1] - self.padding_hauteur):
                for ind_col in range(self.padding_cote, dim_entree_avec_padding[2] - self.padding_cote):
                    patch = entree_avec_padding[:, ind_ligne-self.padding_hauteur:ind_ligne+self.padding_hauteur+1, ind_col-self.padding_cote:ind_col+self.padding_cote+1]
                    mult = patch*filtre.poids
                    mat_image[ind_filtre][ind_ligne - self.padding_hauteur][ind_col - self.padding_cote] = np.sum(mult) + filtre.biais
        return mat_image

class Couche_max_pooling(Couche):
    def __init__(self, dim_filtre):
        self.dim_filtre = dim_filtre
        self.masque = None

    def forward(self, entree):
        assert entree.shape[1] % self.dim_filtre[0] == 0 and entree.shape[2] % self.dim_filtre[1] == 0, "Erreur dans les dimensions du filtre"
        mat_image = np.zeros((entree.shape[0], entree.shape[1]//self.dim_filtre[0], entree.shape[2]//self.dim_filtre[1]))
        self.masque = np.zeros(entree.shape)
        for ind_pro in range(mat_image.shape[0]):
            for ind_ligne in range(mat_image.shape[1]):
                ind_ligne_depart = ind_ligne * self.dim_filtre[0]
                ind_ligne_fin = ind_ligne * self.dim_filtre[0] + self.dim_filtre[0]
                for ind_col in range(mat_image.shape[2]):
                    ind_col_depart = ind_col * self.dim_filtre[1]
                    ind_col_fin = ind_col * self.dim_filtre[1] + self.dim_filtre[1]
                    patch = entree[ind_pro, ind_ligne_depart:ind_ligne_fin, ind_col_depart:ind_col_fin]
                    tuple_indice = np.unravel_index(np.argmax(patch), patch.shape)
                    mat_image[ind_pro, ind_ligne, ind_col] = patch[tuple_indice]
                    self.masque[ind_pro, ind_ligne_depart + tuple_indice[0], ind_col_depart + tuple_indice[1]] = 1
        return mat_image

class Couche_Aplatissement(Couche):
    def forward(self, entree):
        return entree.flatten()

class CoucheDense(Couche):
    def __init__(self, activation, nb_neuronne):
        


    



        
