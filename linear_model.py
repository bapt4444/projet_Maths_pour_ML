import numpy as np

# Fonction softmax
def softmax(scores):
    """
    Transforme les scores en probabilités avec la fonction softmax.

    Paramètre :
    - scores : matrice de taille (n, num_classes)

    Retour :
    - probas : matrice de probabilités de même taille
    """
    exp_scores = np.exp(scores)
    probas = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    return probas


# Fonction de calcul de la Cross-entropy
def cross_entropy_loss(Y_true, probas):
    """
    Calcule la cross-entropy moyenne entre les vrais labels et les probabilités prédites.

    Paramètres :
    - Y_true : matrice des vrais labels au format one-hot, de taille (n, num_classes)
               Chaque ligne correspond à une image, et contient un seul 1 à la position
               de la bonne classe.
    - probas : matrice des probabilités prédites par le modèle, de taille (n, num_classes)
               Chaque ligne correspond à une image, et la somme des probabilités sur une ligne
               vaut 1.

    Retour :
    - logloss : valeur moyenne de la cross-entropy sur l'ensemble des images

    Remarque :
    - On clippe les probabilités avec une très petite valeur epsilon pour éviter de calculer
      log(0), ce qui provoquerait une erreur numérique (voir lien scickit-learn).
    """
    epsilon = 1e-15

    # On borne les probabilités pour éviter log(0)
    probas = np.clip(probas, epsilon, 1 - epsilon)

    # Nombre d'images
    n = Y_true.shape[0]

    # Calcul de la log loss moyenne
    logloss = - np.sum(Y_true * np.log(probas)) / n

    return logloss



def initialize_parameters(input_dim= 784, num_classes=10):
    """
    Role : Initialise les paramètres du modèle linéaire de facon aleatoire.

    Paramètres :
    - input_dim : nombre de pixels en entrée
    - num_classes : nombre de classes

    Retour :
    - A : matrice des poids de dimensions (num_classes lignes , input_dim colonnes)
    - b : vecteur biais de dimensions (1 lignes , num_classes colonnes)
    """
    A = np.random.randn(num_classes, input_dim) * 0.01
    b = np.zeros((1, num_classes))
    return A, b


def compute_scores(X, A, b):
    """
    Role : Calcule les scores du modèle linéaire.

    Paramètres :
    - X : matrice des données de dimensions (n lignes, input_dim colonnes)
    - A : matrice des poids de dimensions (num_classes lignes , input_dim colonnes)
    - b : vecteur biais de dimensions (1 lignes , num_classes colonnes)

    Retour :
    - scores : matrice des scores de taille (n, num_classes)
    """
    scores = X @ A.T + b
    return scores


def compute_gradients(X, Y_true, probas):
    """
    Role : Calcule les gradients de la log loss par rapport aux poids A et au biais b.

    Paramètres :
    - X : matrice des données de dimensions (n lignes, input_dim colonnes)
    - Y_true : matrice des vrais labels au format one-hot, de taille (n, num_classes)
    - probas : matrice des probabilités prédites par le modèle, de taille (n, num_classes)

    Retour :
    - dA : gradient de A, de dimensions (num_classes lignes, input_dim colonnes)
    - db : gradient de b, de dimensions (1 ligne, num_classes colonnes)
    """
    n = X.shape[0]

    dA = ((probas - Y_true).T @ X) / n
    db = np.sum(probas - Y_true, axis=0, keepdims=True) / n

    return dA, db


def update_parameters(A, b, dA, db, learning_rate):
    """
    Role : Met à jour les paramètres du modèle linéaire par descente de gradient.

    Paramètres :
    - A : matrice des poids
    - b : vecteur biais
    - dA : gradient de A
    - db : gradient de b
    - learning_rate : pas d'apprentissage

    Retour :
    - A : matrice des poids mise à jour
    - b : vecteur biais mis à jour
    """
    A = A - learning_rate * dA
    b = b - learning_rate * db
    return A, b


def train_linear_model(X, Y_true, input_dim=784, num_classes=10, learning_rate=0.1, epochs=100):
    """
    Role : Entraîne le modèle linéaire par descente de gradient.

    Paramètres :
    - X : matrice des données de dimensions (n lignes, input_dim colonnes)
    - Y_true : matrice des vrais labels au format one-hot et de dimensions (n, num_classes)
    - input_dim : nombre de pixels en entrée
    - num_classes : nombre de classes
    - learning_rate : pas d'apprentissage
    - epochs : nombre d'epochs

    Retour :
    - A : matrice des poids entraînée
    - b : vecteur biais entraîné
    - loss_history : liste contenant la log loss à chaque époque
    """
    A, b = initialize_parameters(input_dim=input_dim, num_classes=num_classes)

    loss_history = []

    for epoch in range(epochs):
        # 1) calcul des scores
        scores = compute_scores(X, A, b)

        # 2) calcul des probabilités
        probas = softmax(scores)

        # 3) calcul de la loss
        logloss = cross_entropy_loss(Y_true, probas)
        loss_history.append(logloss)

        # 4) calcul des gradients
        dA, db = compute_gradients(X, Y_true, probas)

        # 5) mise à jour des paramètres
        A, b = update_parameters(A, b, dA, db, learning_rate)

    return A, b, loss_history