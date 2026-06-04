import numpy as np


def softmax(scores):
    # On soustrait le max pour la stabilité numérique
    shift_scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shift_scores)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)



def cross_entropy_loss(Y_true, probas):
    epsilon = 1e-15
    probas = np.clip(probas, epsilon, 1 - epsilon)
    n = Y_true.shape[0]
    return - np.sum(Y_true * np.log(probas)) / n


def initialize_parameters(input_dim=784, num_classes=10):
    """
    On utilise l'initialisation de Xavier :
    L'écart-type est sqrt(1 / input_dim).
    Cela permet de garder une variance stable à travers le réseau.
    """
    limit = np.sqrt(1 / input_dim)
    A = np.random.randn(num_classes, input_dim) * limit
    b = np.zeros((1, num_classes))
    return A, b


def compute_scores(X, A, b):
    return X @ A.T + b


def compute_gradients(X, Y_true, probas):
    n = X.shape[0]
    dA = ((probas - Y_true).T @ X) / n
    db = np.sum(probas - Y_true, axis=0, keepdims=True) / n
    return dA, db


def update_parameters(A, b, dA, db, learning_rate):
    A = A - learning_rate * dA
    b = b - learning_rate * db
    return A, b


def train_linear_model(X, Y_true, input_dim=784, num_classes=10, learning_rate=0.1, epochs=300):
    # Normalisation CRUCIALE si pas faite avant
    if np.max(X) > 1.0:
        X = X / 255.0

    A, b = initialize_parameters(input_dim, num_classes)
    loss_history = []

    for epoch in range(epochs):
        probas = softmax(compute_scores(X, A, b))
        loss = cross_entropy_loss(Y_true, probas)
        loss_history.append(loss)

        dA, db = compute_gradients(X, Y_true, probas)
        A, b = update_parameters(A, b, dA, db, learning_rate)

        # Affichage tous les 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    return A, b, loss_history


def predict(X, A, b):
    """
    Même logique que le modèle 1 :
    Calcul des scores -> Softmax -> Argmax
    """
    scores = X @ A.T + b
    # Utilise le softmax stable que tu as normalement mis dans ce fichier
    probas = softmax(scores)
    return np.argmax(probas, axis=1)

def accuracy(y_true, y_pred):
    """
    Calcule la proportion de bonnes prédictions.
    """
    return np.mean(y_true == y_pred)
