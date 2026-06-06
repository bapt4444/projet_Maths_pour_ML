import cupy as np
import time
import pickle

from couche import (Couche_convolution, Couche_max_pooling,
                    Couche_Aplatissement, CoucheDense)


class Modele:
    def __init__(self):
        self.tab_couche = []
        self.historique = {"loss_train": [], "acc_train": [],
                           "loss_test": [], "acc_test": []}



    def ajouter_couche_conv(self, nb_filtre, dim_filtre, activation, n_entree=None):
        if n_entree is None and self.tab_couche:
            n_entree = self.tab_couche[-1].dim_sortie
        self.tab_couche.append(Couche_convolution(nb_filtre, dim_filtre,
                                                  activation, n_entree))

    def ajouter_couche_max_pooling(self, dim_filtre, n_entree=None):
        if n_entree is None and self.tab_couche:
            n_entree = self.tab_couche[-1].dim_sortie
        self.tab_couche.append(Couche_max_pooling(dim_filtre, n_entree))

    def ajouter_couche_applatissement(self, n_entree=None):
        if n_entree is None and self.tab_couche:
            n_entree = self.tab_couche[-1].dim_sortie
        self.tab_couche.append(Couche_Aplatissement(n_entree))

    def ajouter_couche_dense(self, activation, nb_neurones, n_entree=None):
        if n_entree is None and self.tab_couche:
            n_entree = self.tab_couche[-1].dim_sortie
        self.tab_couche.append(CoucheDense(activation, n_entree, nb_neurones))



    def forward(self, batch):
        for couche in self.tab_couche:
            batch = couche.forward(batch)
        return batch

    def backward(self, y_pred, y_true, learning_rate):
        # Gradient combine Softmax + CrossEntropy : y_pred - y_true
        gradient = (y_pred - y_true) / y_true.shape[0]
        for couche in reversed(self.tab_couche):
            gradient = couche.backward(gradient, learning_rate)



    @staticmethod
    def _cross_entropy(y_pred, y_true, eps=1e-12):
        return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))

    @staticmethod
    def _accuracy(y_pred, y_true):
        return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))

    def evaluer(self, X, Y, batch_size=128):
        """Evaluation par batchs pour eviter de tout charger en memoire."""
        n = X.shape[0]
        loss_tot, acc_tot, vus = 0.0, 0.0, 0
        for debut in range(0, n, batch_size):
            fin = min(debut + batch_size, n)
            xb, yb = X[debut:fin], Y[debut:fin]
            pred = self.forward(xb)
            loss_tot += self._cross_entropy(pred, yb) * (fin - debut)
            acc_tot += self._accuracy(pred, yb) * (fin - debut)
            vus += (fin - debut)
        return loss_tot / vus, acc_tot / vus



    def entrainer(self, X_train, Y_train, epochs, learning_rate,
                  batch_size=64, X_test=None, Y_test=None, shuffle=True,
                  verbose=True):
        n = X_train.shape[0]
        nb_batchs = (n + batch_size - 1) // batch_size

        for e in range(epochs):
            t0 = time.time()
            if shuffle:
                ordre = np.random.permutation(n)
                X_train = X_train[ordre]
                Y_train = Y_train[ordre]

            loss_acc, acc_acc, vus = 0.0, 0.0, 0
            for b in range(nb_batchs):
                debut = b * batch_size
                fin = min(debut + batch_size, n)
                xb = X_train[debut:fin]
                yb = Y_train[debut:fin]

                pred = self.forward(xb)
                self.backward(pred, yb, learning_rate)


                loss_acc += self._cross_entropy(pred, yb) * (fin - debut)
                acc_acc += self._accuracy(pred, yb) * (fin - debut)
                vus += (fin - debut)

                if verbose and (b % 50 == 0 or b == nb_batchs - 1):
                    print(f"  epoch {e+1}/{epochs} - batch {b+1}/{nb_batchs} "
                          f"- loss {loss_acc/vus:.4f} - acc {acc_acc/vus:.4f}",
                          end="\r")

            loss_train = loss_acc / vus
            acc_train = acc_acc / vus
            self.historique["loss_train"].append(loss_train)
            self.historique["acc_train"].append(acc_train)

            msg = (f"Epoch {e+1}/{epochs} - {time.time()-t0:.1f}s "
                   f"- loss {loss_train:.4f} - acc {acc_train:.4f}")

            if X_test is not None and Y_test is not None:
                loss_test, acc_test = self.evaluer(X_test, Y_test, batch_size)
                self.historique["loss_test"].append(loss_test)
                self.historique["acc_test"].append(acc_test)
                msg += f" - val_loss {loss_test:.4f} - val_acc {acc_test:.4f}"

            print(" " * 100, end="\r")
            print(msg)

        return self.historique
    
    def sauvegarder(self, chemin_fichier="mon_modele.pkl"):
        with open(chemin_fichier, 'wb') as fichier:
            pickle.dump(self, fichier)
        print(f"Modèle sauvegardé avec succès dans : {chemin_fichier}")

    @classmethod
    def charger(cls, chemin_fichier="mon_modele.pkl"):

        with open(chemin_fichier, 'rb') as fichier:
            modele = pickle.load(fichier)
        print(f"Modèle chargé avec succès depuis : {chemin_fichier}")
        return modele
