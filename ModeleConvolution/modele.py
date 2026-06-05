from couche import *

class modele():
    def __init__(self):
        self.tab_couche = []
    
    def ajouter_couche_conv(self, nb_filtre, dim_filtre, activation, n_entree=None):
        couche = Couche_convolution(nb_filtre, dim_filtre, activation, )
        self.tab_couche.append(couche)
    
    def ajouter_couche_max_pooling(self, dim_filtre,n_entree=None):
        if n_entree is None and len(self.tab_couche) != 0:
            n_entree = self.tab_couche[-1].dim_sortie
        couche = Couche_max_pooling(dim_filtre,n_entree)
        self.tab_couche.append(couche)
    
    def ajouter_couche_applatissement(self, n_entree=None):
        n_entree = self.tab_couche[-1].dim_sortie
        couche = Couche_Aplatissement(n_entree)
        self.tab_couche.append(couche)
    
    def couche_dense(self, activation, nb_neuronne, n_entree=None):
        couche = CoucheDense(activation, nb_neuronne, n_entree)
    
        
        
    