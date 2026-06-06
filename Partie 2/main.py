import os
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count())

import cupy as np
np.set_printoptions(suppress=True)

from tensorflow.keras.datasets import cifar10

from modele import Modele
from activation import ReLU, Softmax
from utils import one_hot_encode


(X_train, y_train), (X_test, y_test) = cifar10.load_data()

class_names = ["avion", "automobile", "oiseau", "chat", "cerf",
               "chien", "grenouille", "cheval", "bateau", "camion"]

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
X_train = np.transpose(X_train, (0, 3, 1, 2)).astype(np.float32)
X_test = np.transpose(X_test, (0, 3, 1, 2)).astype(np.float32)
mean = X_train.mean(axis=(0, 2, 3), keepdims=True)
std = X_train.std(axis=(0, 2, 3), keepdims=True)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

Y_train = one_hot_encode(y_train, num_classes=10)
Y_test = one_hot_encode(y_test, num_classes=10)
Y_train = np.asarray(Y_train)
Y_test = np.asarray(Y_test)

print(f"X_train : {X_train.shape} - X_test : {X_test.shape}")



#  Architecture:
#  Entree     (3, 32, 32)
#  Conv 64    (64, 32, 32)
#  Conv3D 64  (64, 32, 32)
#  MaxPool    (64, 16, 16)
#  Conv3D 64  (64, 16, 16)
#  MaxPool    (64,  8,  8)
#  Conv3D 64  (64,  8,  8)
#  Flatten    (4096,)
#  Dense 10   (10,)  +  Softmax


modele = Modele()
modele.ajouter_couche_conv(64, (3, 3, 3), ReLU, n_entree=(3, 32, 32))
modele.ajouter_couche_conv(64, (64, 3, 3), ReLU)
modele.ajouter_couche_max_pooling((2, 2))
modele.ajouter_couche_conv(64, (64, 3, 3), ReLU)
modele.ajouter_couche_max_pooling((2, 2))
modele.ajouter_couche_conv(64, (64, 3, 3), ReLU)
modele.ajouter_couche_applatissement()
modele.ajouter_couche_dense(Softmax, 10)


print(modele.historique)
modele.entrainer(X_train, Y_train,
                 epochs=60,
                 learning_rate=0.05,
                 batch_size=128,
                 X_test=X_test, Y_test=Y_test,
                 shuffle=True,
                 verbose=True)


modele.sauvegarder(chemin_fichier="test2.pkl")
loss_test, acc_test = modele.evaluer(X_test, Y_test, batch_size=128)
print(f"\nResultats finaux sur le test set :")
print(f"  loss = {loss_test:.4f}")
print(f"  acc  = {acc_test:.4f}")
print(f"  taux d'erreur = {(1-acc_test)*100:.2f} %")
