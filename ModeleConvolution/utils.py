import numpy as np



## Fonction d'encodage des labels
def one_hot_encode(labels, num_classes=10):
    """
    Transforme une liste de labels entiers en matrice one-hot.

    Paramètres :
    - labels : liste contenant les labels
    - num_classes : nombre de classes a encoder

    Retour :
    - Y : matrice où chaque ligne correspond à un label encodé en one-hot
    """
    # conversion en np.array au cas ou
    labels = np.array(labels, dtype=int)
    n = len(labels)
    Y = np.zeros((n, num_classes))

    for i in range(n):
        Y[i, labels[i]] = 1

    return Y