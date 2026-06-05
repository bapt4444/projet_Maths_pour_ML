from couche import *

class Modele():
    def __init__(self):
        self.tab_couche = []
    
    def ajouter_couche_conv(self, nb_filtre, dim_filtre, activation, n_entree=None):
        if n_entree is None and len(self.tab_couche) != 0:
            n_entree = self.tab_couche[-1].dim_sortie
        couche = Couche_convolution(nb_filtre, dim_filtre, activation, n_entree)
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
    
    def ajouter_couche_dense(self, activation, nb_neuronne, n_entree=None):
        if n_entree is None and len(self.tab_couche) != 0:
            n_entree = self.tab_couche[-1].dim_sortie
        couche = CoucheDense(activation, n_entree, nb_neuronne)
        self.tab_couche.append(couche)
    
    def forward(self, image):
        for couche in self.tab_couche:
            image = couche.forward(image)
        return image
    
    def backward(self, y_pred, y_true, learning_rate):
        gradient = y_pred - y_true
        for couche in reversed(self.tab_couche):
            gradient = couche.backward(gradient, learning_rate)
    
    def entrainer(self, x_train, y_train, epochs, learning_rate):
        for e in range(epochs):
            i = 0
            for image, etiquette_vraie in zip(x_train, y_train):
                print(f"epoch : {e}, image : {i}")
                i += 1
                prediction = self.forward(image)
                self.backward(prediction, etiquette_vraie, learning_rate)


    
        
        
    