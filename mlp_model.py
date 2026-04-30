import numpy as np


# =========================
# Fonctions utiles
# =========================

def softmax(scores):
    """
    Transforme les scores en probabilités avec softmax.

    Paramètres :
    - scores : matrice de taille (n, num_classes)

    Retour :
    - probas : matrice de probabilités de même taille
    """
    # Version stable pour éviter les problèmes numériques avec exp
    shifted_scores = scores - np.max(scores, axis=1, keepdims=True)

    exp_scores = np.exp(shifted_scores)
    probas = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    return probas


def cross_entropy_loss(Y_true, probas):
    """
    Calcule la cross-entropy moyenne.

    Paramètres :
    - Y_true : vrais labels au format one-hot, taille (n, num_classes)
    - probas : probabilités prédites, taille (n, num_classes)

    Retour :
    - logloss : valeur moyenne de la loss
    """
    epsilon = 1e-15

    # On évite log(0)
    probas = np.clip(probas, epsilon, 1 - epsilon)

    n = Y_true.shape[0]

    logloss = -np.sum(Y_true * np.log(probas)) / n

    return logloss


def relu(x):
    """
    Fonction d'activation ReLU.
    """
    return np.maximum(0, x)


def relu_derivative(x):
    """
    Dérivée de ReLU.

    Pour ReLU :
    - dérivée = 1 si x > 0
    - dérivée = 0 sinon
    """
    return (x > 0).astype(float)


def accuracy(y_true, y_pred):
    """
    Calcule l'accuracy du modèle.
    """
    return np.mean(y_true == y_pred)


# =========================
# MLP à une couche cachée
# =========================

def initialize_parameters_1_hidden(input_dim=784, hidden_dim=128, num_classes=10):
    """
    Initialise les paramètres d'un MLP à une couche cachée.

    Paramètres :
    - input_dim : nombre de pixels en entrée
    - hidden_dim : nombre de neurones dans la couche cachée
    - num_classes : nombre de classes en sortie

    Retour :
    - A1 : poids de la couche cachée, taille (hidden_dim, input_dim)
    - b1 : biais de la couche cachée, taille (1, hidden_dim)
    - A2 : poids de la couche de sortie, taille (num_classes, hidden_dim)
    - b2 : biais de la couche de sortie, taille (1, num_classes)
    """
    # Initialisation adaptée à ReLU
    A1 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2 / input_dim)
    b1 = np.zeros((1, hidden_dim))

    A2 = np.random.randn(num_classes, hidden_dim) * np.sqrt(1 / hidden_dim)
    b2 = np.zeros((1, num_classes))

    return A1, b1, A2, b2


def forward_1_hidden(X, A1, b1, A2, b2):
    """
    Propagation avant pour un MLP à une couche cachée.

    Retour :
    - probas : probabilités prédites
    - cache : valeurs intermédiaires utiles pour la rétropropagation
    """
    # Première couche cachée
    O1 = X @ A1.T + b1
    Z1 = relu(O1)

    # Couche de sortie
    scores = Z1 @ A2.T + b2
    probas = softmax(scores)

    cache = {
        "X": X,
        "O1": O1,
        "Z1": Z1,
        "scores": scores,
        "probas": probas
    }

    return probas, cache


def compute_gradients_1_hidden(Y_true, A2, cache):
    """
    Calcule les gradients pour un MLP à une couche cachée.

    On utilise les formules :

    dL/da2(k,q) = 1/n sum_i z1(i,q) (P(i,k) - y_i(k))

    dL/da1(q,m) = 1/n sum_i x(i,m) phi'(o1(i,q))
                  [sum_k a2(k,q)(P(i,k)-y_i(k))]
    """
    X = cache["X"]
    O1 = cache["O1"]
    Z1 = cache["Z1"]
    probas = cache["probas"]

    n = X.shape[0]

    # Erreur de sortie :
    # dE_i / d o_i,k = P_i,k - y_i(k)
    dScores = probas - Y_true

    # Gradients de la couche de sortie
    dA2 = (dScores.T @ Z1) / n
    db2 = np.sum(dScores, axis=0, keepdims=True) / n

    # Erreur ramenée vers la couche cachée
    dZ1 = dScores @ A2

    # Passage dans la dérivée de l'activation
    dO1 = dZ1 * relu_derivative(O1)

    # Gradients de la première couche cachée
    dA1 = (dO1.T @ X) / n
    db1 = np.sum(dO1, axis=0, keepdims=True) / n

    return dA1, db1, dA2, db2


def update_parameters_1_hidden(A1, b1, A2, b2, dA1, db1, dA2, db2, learning_rate):
    """
    Met à jour les paramètres par descente de gradient.
    """
    A1 = A1 - learning_rate * dA1
    b1 = b1 - learning_rate * db1

    A2 = A2 - learning_rate * dA2
    b2 = b2 - learning_rate * db2

    return A1, b1, A2, b2


def train_mlp_1_hidden(X, Y_true, input_dim=784, hidden_dim=128, num_classes=10,
                       learning_rate=0.1, epochs=100):
    """
    Entraîne un MLP à une couche cachée.
    """
    A1, b1, A2, b2 = initialize_parameters_1_hidden(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes
    )

    loss_history = []

    for epoch in range(epochs):
        # 1) Forward
        probas, cache = forward_1_hidden(X, A1, b1, A2, b2)

        # 2) Loss
        loss = cross_entropy_loss(Y_true, probas)
        loss_history.append(loss)

        # 3) Backward
        dA1, db1, dA2, db2 = compute_gradients_1_hidden(Y_true, A2, cache)

        # 4) Update
        A1, b1, A2, b2 = update_parameters_1_hidden(
            A1, b1, A2, b2,
            dA1, db1, dA2, db2,
            learning_rate
        )

    return A1, b1, A2, b2, loss_history


def predict_1_hidden(X, A1, b1, A2, b2):
    """
    Prédit les classes avec le MLP à une couche cachée.
    """
    probas, _ = forward_1_hidden(X, A1, b1, A2, b2)
    y_pred = np.argmax(probas, axis=1)

    return y_pred


# =========================
# MLP à deux couches cachées
# =========================

def initialize_parameters_2_hidden(input_dim=784, hidden_dim1=128, hidden_dim2=64, num_classes=10):
    """
    Initialise les paramètres d'un MLP à deux couches cachées.

    Retour :
    - A1, b1 : paramètres de la première couche cachée
    - A2, b2 : paramètres de la deuxième couche cachée
    - A3, b3 : paramètres de la couche de sortie
    """
    A1 = np.random.randn(hidden_dim1, input_dim) * np.sqrt(2 / input_dim)
    b1 = np.zeros((1, hidden_dim1))

    A2 = np.random.randn(hidden_dim2, hidden_dim1) * np.sqrt(2 / hidden_dim1)
    b2 = np.zeros((1, hidden_dim2))

    A3 = np.random.randn(num_classes, hidden_dim2) * np.sqrt(1 / hidden_dim2)
    b3 = np.zeros((1, num_classes))

    return A1, b1, A2, b2, A3, b3


def forward_2_hidden(X, A1, b1, A2, b2, A3, b3):
    """
    Propagation avant pour un MLP à deux couches cachées.
    """
    # Première couche cachée
    O1 = X @ A1.T + b1
    Z1 = relu(O1)

    # Deuxième couche cachée
    O2 = Z1 @ A2.T + b2
    Z2 = relu(O2)

    # Couche de sortie
    scores = Z2 @ A3.T + b3
    probas = softmax(scores)

    cache = {
        "X": X,
        "O1": O1,
        "Z1": Z1,
        "O2": O2,
        "Z2": Z2,
        "scores": scores,
        "probas": probas
    }

    return probas, cache


def compute_gradients_2_hidden(Y_true, A2, A3, cache):
    """
    Calcule les gradients pour un MLP à deux couches cachées.

    Formules utilisées :

    Couche de sortie :
    dL/da3(k,t) = 1/n sum_i z2(i,t)(P(i,k)-y_i(k))

    Deuxième couche cachée :
    dL/da2(t,q) = 1/n sum_i z1(i,q) phi2'(o2(i,t))
                  [sum_k a3(k,t)(P(i,k)-y_i(k))]

    Première couche cachée :
    dL/da1(q,m) = 1/n sum_i x(i,m) phi1'(o1(i,q))
                  [sum_t a2(t,q) phi2'(o2(i,t))
                  [sum_k a3(k,t)(P(i,k)-y_i(k))]]
    """
    X = cache["X"]
    O1 = cache["O1"]
    Z1 = cache["Z1"]
    O2 = cache["O2"]
    Z2 = cache["Z2"]
    probas = cache["probas"]

    n = X.shape[0]

    # Erreur de sortie :
    # dE_i / d o_i,k = P_i,k - y_i(k)
    dScores = probas - Y_true

    # =========================
    # Gradients couche de sortie
    # =========================
    dA3 = (dScores.T @ Z2) / n
    db3 = np.sum(dScores, axis=0, keepdims=True) / n

    # =========================
    # Rétropropagation vers couche cachée 2
    # =========================
    # dZ2 correspond à l'erreur ramenée vers Z2
    dZ2 = dScores @ A3

    # dO2 = erreur du neurone t de la couche 2
    dO2 = dZ2 * relu_derivative(O2)

    # Gradient des poids A2
    dA2 = (dO2.T @ Z1) / n
    db2 = np.sum(dO2, axis=0, keepdims=True) / n

    # =========================
    # Rétropropagation vers couche cachée 1
    # =========================
    # dZ1 correspond à l'erreur ramenée vers Z1
    dZ1 = dO2 @ A2

    # dO1 = erreur du neurone q de la couche 1
    dO1 = dZ1 * relu_derivative(O1)

    # Gradient des poids A1
    dA1 = (dO1.T @ X) / n
    db1 = np.sum(dO1, axis=0, keepdims=True) / n

    return dA1, db1, dA2, db2, dA3, db3


def update_parameters_2_hidden(A1, b1, A2, b2, A3, b3,
                               dA1, db1, dA2, db2, dA3, db3,
                               learning_rate):
    """
    Met à jour les paramètres du MLP à deux couches cachées.
    """
    A1 = A1 - learning_rate * dA1
    b1 = b1 - learning_rate * db1

    A2 = A2 - learning_rate * dA2
    b2 = b2 - learning_rate * db2

    A3 = A3 - learning_rate * dA3
    b3 = b3 - learning_rate * db3

    return A1, b1, A2, b2, A3, b3


def train_mlp_2_hidden(X, Y_true, input_dim=784, hidden_dim1=128, hidden_dim2=64,
                       num_classes=10, learning_rate=0.1, epochs=100):
    """
    Entraîne un MLP à deux couches cachées.
    """
    A1, b1, A2, b2, A3, b3 = initialize_parameters_2_hidden(
        input_dim=input_dim,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        num_classes=num_classes
    )

    loss_history = []

    for epoch in range(epochs):
        # 1) Forward
        probas, cache = forward_2_hidden(X, A1, b1, A2, b2, A3, b3)

        # 2) Loss
        loss = cross_entropy_loss(Y_true, probas)
        loss_history.append(loss)

        # 3) Backward
        dA1, db1, dA2, db2, dA3, db3 = compute_gradients_2_hidden(
            Y_true, A2, A3, cache
        )

        # 4) Update
        A1, b1, A2, b2, A3, b3 = update_parameters_2_hidden(
            A1, b1, A2, b2, A3, b3,
            dA1, db1, dA2, db2, dA3, db3,
            learning_rate
        )

    return A1, b1, A2, b2, A3, b3, loss_history


def predict_2_hidden(X, A1, b1, A2, b2, A3, b3):
    """
    Prédit les classes avec le MLP à deux couches cachées.
    """
    probas, _ = forward_2_hidden(X, A1, b1, A2, b2, A3, b3)
    y_pred = np.argmax(probas, axis=1)

    return y_pred