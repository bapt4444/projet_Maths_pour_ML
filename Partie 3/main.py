import os
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count())

import cupy as np
np.set_printoptions(suppress=True)



from modele import Modele
from activation import ReLU, Softmax
from utils import one_hot_encode


(X_train, y_train), (X_test, y_test) = None

class_names = ["MALIGNANT", "BENIGN_WITHOUT_CALLBACK","BENIGN"]



Y_train = one_hot_encode(y_train, num_classes=10)
Y_test = one_hot_encode(y_test, num_classes=10)
Y_train = np.asarray(Y_train)
Y_test = np.asarray(Y_test)

print(f"X_train : {X_train.shape} - X_test : {X_test.shape}")



#  Architecture:
#  Entree     (1, 128, 128)
#  Conv 64    (64, 128, 128)
#  MaxPool    (64, 64, 64)
#  Conv3D 64  (64, 64, 64)
#  MaxPool    (64,  32,  32)
#  Conv3D 64  (64,  32,  32)
#  MaxPool    (64,  16,  16)
#  Conv3D 64  (64,  16,  16)
#  Flatten    (16384)
#  Dense 10   (3,)  +  Softmax


modele = Modele()
modele.ajouter_couche_conv(64, (1, 3, 3), ReLU, n_entree=(1, 128, 128))
modele.ajouter_couche_conv(64, (64, 3, 3), ReLU)
modele.ajouter_couche_max_pooling((2, 2))
modele.ajouter_couche_conv(64, (64, 3, 3), ReLU)
modele.ajouter_couche_max_pooling((2, 2))
modele.ajouter_couche_conv(64, (64, 3, 3), ReLU)
modele.ajouter_couche_max_pooling((2, 2))
modele.ajouter_couche_conv(64, (64, 3, 3), ReLU)
modele.ajouter_couche_applatissement()
modele.ajouter_couche_dense(Softmax, 3)


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
