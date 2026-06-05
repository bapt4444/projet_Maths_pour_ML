from filtre import *
from modele import *
from activation import *
from tensorflow.keras.datasets import cifar10
from utils import *

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
class_names = [
    "avion",
    "automobile",
    "oiseau",
    "chat",
    "cerf",
    "chien",
    "grenouille",
    "cheval",
    "bateau",
    "camion"
]
Y_train = one_hot_encode(y_train, num_classes=10)
Y_test = one_hot_encode(y_test, num_classes=10)
X_train = np.transpose(X_train, (0, 3, 1, 2))
X_test = np.transpose(X_test, (0, 3, 1, 2))
Modele_test = Modele()
Modele_test.ajouter_couche_conv(64, (3,3,3), ReLU, n_entree=(3,32,32))
Modele_test.ajouter_couche_conv(64, (64,3,3), ReLU)
Modele_test.ajouter_couche_max_pooling((2,2))
Modele_test.ajouter_couche_conv(64, (64,3,3), ReLU)
Modele_test.ajouter_couche_max_pooling((2,2))
Modele_test.ajouter_couche_conv(64, (64,3,3), ReLU)
Modele_test.ajouter_couche_applatissement()
Modele_test.ajouter_couche_dense(Softmax, 10)
Modele_test.entrainer(X_train, Y_train, 10, 0.0001)



