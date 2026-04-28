import numpy as np


def softmax(scores):
    """
    Transforme les scores en probabilités avec une version numériquement stable.

    Paramètres :
    - scores : matrice de taille (n, num_classes)

    Retour :
    - probas : matrice de probabilités de même taille
    """
    shifted_scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted_scores)
    probas = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probas


def cross_entropy_loss(Y_true, probas):
    """
    Calcule la cross-entropy moyenne entre les vrais labels et les probabilités prédites.

    Paramètres :
    - Y_true : matrice des vrais labels au format one-hot, de taille (n, num_classes)
    - probas : matrice des probabilités prédites, de taille (n, num_classes)

    Retour :
    - logloss : valeur moyenne de la cross-entropy
    """
    epsilon = 1e-15
    probas = np.clip(probas, epsilon, 1 - epsilon)
    n = Y_true.shape[0]
    logloss = -np.sum(Y_true * np.log(probas)) / n
    return logloss


def relu(x):
    """Applique la fonction d'activation ReLU."""
    return np.maximum(0, x)


def relu_derivative(x):
    """Retourne la dérivée de ReLU."""
    return (x > 0).astype(float)


def sigmoid(x):
    """Applique la fonction sigmoïde."""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """Retourne la dérivée de la sigmoïde."""
    sig = sigmoid(x)
    return sig * (1 - sig)


def tanh(x):
    """Applique la tangente hyperbolique."""
    return np.tanh(x)


def tanh_derivative(x):
    """Retourne la dérivée de tanh."""
    return 1 - np.tanh(x) ** 2


def get_activation_functions(name):
    """
    Sélectionne la fonction d'activation cachée et sa dérivée.

    Paramètres :
    - name : nom de l'activation ('relu', 'sigmoid' ou 'tanh')

    Retour :
    - activation_fn : fonction d'activation
    - derivative_fn : dérivée de la fonction d'activation
    """
    activations = {
        "relu": (relu, relu_derivative),
        "sigmoid": (sigmoid, sigmoid_derivative),
        "tanh": (tanh, tanh_derivative),
    }

    if name not in activations:
        raise ValueError(
            "Activation inconnue. Choisir parmi : 'relu', 'sigmoid', 'tanh'."
        )

    return activations[name]


def normalize_hidden_dims(hidden_dims):
    """
    Normalise la description des couches cachées.

    Paramètres :
    - hidden_dims : entier ou liste/tuple d'entiers

    Retour :
    - hidden_dims : liste d'entiers
    """
    if isinstance(hidden_dims, int):
        hidden_dims = [hidden_dims]
    elif isinstance(hidden_dims, tuple):
        hidden_dims = list(hidden_dims)
    elif not isinstance(hidden_dims, list):
        raise TypeError("hidden_dims doit être un entier, une liste ou un tuple.")

    if len(hidden_dims) == 0:
        raise ValueError("Le modèle MLP doit contenir au moins une couche cachée.")

    if any(dim <= 0 for dim in hidden_dims):
        raise ValueError("Toutes les tailles de couches cachées doivent être positives.")

    return hidden_dims


def initialize_mlp_parameters(input_dim, hidden_dims, num_classes, activation="relu", seed=None):
    """
    Initialise les paramètres d'un réseau de neurones multi-couches.

    Paramètres :
    - input_dim : dimension de l'entrée
    - hidden_dims : liste des tailles des couches cachées
    - num_classes : nombre de classes en sortie
    - activation : activation cachée utilisée, afin d'adapter l'initialisation
    - seed : graine aléatoire optionnelle

    Retour :
    - parameters : dictionnaire contenant W1, b1, ..., WL, bL
    """
    hidden_dims = normalize_hidden_dims(hidden_dims)

    if seed is not None:
        np.random.seed(seed)

    layer_dims = [input_dim] + hidden_dims + [num_classes]
    parameters = {}

    for layer_idx in range(1, len(layer_dims)):
        fan_in = layer_dims[layer_idx - 1]
        fan_out = layer_dims[layer_idx]

        if layer_idx < len(layer_dims) - 1 and activation == "relu":
            scale = np.sqrt(2 / fan_in)
        else:
            scale = np.sqrt(1 / fan_in)

        parameters[f"W{layer_idx}"] = np.random.randn(fan_out, fan_in) * scale
        parameters[f"b{layer_idx}"] = np.zeros((1, fan_out))

    return parameters


def forward_mlp(X, parameters, activation="relu"):
    """
    Effectue la propagation avant dans le réseau.

    Paramètres :
    - X : matrice des données de taille (n, input_dim)
    - parameters : dictionnaire des poids et biais
    - activation : fonction d'activation des couches cachées

    Retour :
    - probas : probabilités de sortie après softmax
    - cache : informations intermédiaires utiles pour la rétropropagation
    """
    activation_fn, _ = get_activation_functions(activation)
    num_layers = len(parameters) // 2

    cache = {
        "A0": X,
    }
    current_activation = X

    for layer_idx in range(1, num_layers):
        W = parameters[f"W{layer_idx}"]
        b = parameters[f"b{layer_idx}"]
        Z = current_activation @ W.T + b
        A = activation_fn(Z)

        cache[f"Z{layer_idx}"] = Z
        cache[f"A{layer_idx}"] = A
        current_activation = A

    W_out = parameters[f"W{num_layers}"]
    b_out = parameters[f"b{num_layers}"]
    scores = current_activation @ W_out.T + b_out
    probas = softmax(scores)

    cache[f"Z{num_layers}"] = scores
    cache[f"A{num_layers}"] = probas

    return probas, cache


def backward_mlp(Y_true, parameters, cache, activation="relu"):
    """
    Effectue la rétropropagation dans le réseau.

    Paramètres :
    - Y_true : matrice one-hot des vraies classes
    - parameters : dictionnaire des poids et biais
    - cache : résultats intermédiaires de la propagation avant
    - activation : fonction d'activation des couches cachées

    Retour :
    - gradients : dictionnaire contenant les gradients de W1, b1, ..., WL, bL
    """
    _, derivative_fn = get_activation_functions(activation)
    num_layers = len(parameters) // 2
    n = Y_true.shape[0]
    gradients = {}

    dZ = cache[f"A{num_layers}"] - Y_true

    for layer_idx in range(num_layers, 0, -1):
        A_prev = cache[f"A{layer_idx - 1}"]
        W = parameters[f"W{layer_idx}"]

        gradients[f"dW{layer_idx}"] = (dZ.T @ A_prev) / n
        gradients[f"db{layer_idx}"] = np.sum(dZ, axis=0, keepdims=True) / n

        if layer_idx > 1:
            dA_prev = dZ @ W
            dZ = dA_prev * derivative_fn(cache[f"Z{layer_idx - 1}"])

    return gradients


def update_mlp_parameters(parameters, gradients, learning_rate):
    """
    Met à jour les paramètres du réseau par descente de gradient.

    Paramètres :
    - parameters : dictionnaire des poids et biais
    - gradients : dictionnaire des gradients
    - learning_rate : pas d'apprentissage

    Retour :
    - parameters : dictionnaire mis à jour
    """
    num_layers = len(parameters) // 2

    for layer_idx in range(1, num_layers + 1):
        parameters[f"W{layer_idx}"] -= learning_rate * gradients[f"dW{layer_idx}"]
        parameters[f"b{layer_idx}"] -= learning_rate * gradients[f"db{layer_idx}"]

    return parameters


def train_mlp_model(
    X,
    Y_true,
    hidden_dims,
    learning_rate=0.1,
    epochs=100,
    activation="relu",
    seed=None,
    verbose=False,
):
    """
    Entraîne un réseau de neurones multi-couches par descente de gradient.

    Paramètres :
    - X : matrice des données de taille (n, input_dim)
    - Y_true : labels au format one-hot, de taille (n, num_classes)
    - hidden_dims : taille d'une couche cachée ou liste des tailles des couches cachées
    - learning_rate : pas d'apprentissage
    - epochs : nombre d'époques
    - activation : activation cachée ('relu', 'sigmoid' ou 'tanh')
    - seed : graine aléatoire optionnelle
    - verbose : affiche la loss périodiquement si True

    Retour :
    - parameters : paramètres entraînés
    - loss_history : historique de la loss
    """
    hidden_dims = normalize_hidden_dims(hidden_dims)
    input_dim = X.shape[1]
    num_classes = Y_true.shape[1]

    parameters = initialize_mlp_parameters(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        activation=activation,
        seed=seed,
    )

    loss_history = []

    for epoch in range(epochs):
        probas, cache = forward_mlp(X, parameters, activation=activation)
        logloss = cross_entropy_loss(Y_true, probas)
        loss_history.append(logloss)

        gradients = backward_mlp(Y_true, parameters, cache, activation=activation)
        parameters = update_mlp_parameters(parameters, gradients, learning_rate)

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch + 1}/{epochs} - loss : {logloss:.6f}")

    return parameters, loss_history


def predict_proba_mlp(X, parameters, activation="relu"):
    """
    Retourne les probabilités prédites par le réseau.

    Paramètres :
    - X : matrice des données
    - parameters : paramètres entraînés
    - activation : activation cachée

    Retour :
    - probas : probabilités prédites
    """
    probas, _ = forward_mlp(X, parameters, activation=activation)
    return probas


def predict_mlp(X, parameters, activation="relu"):
    """
    Prédit la classe de chaque entrée.

    Paramètres :
    - X : matrice des données
    - parameters : paramètres entraînés
    - activation : activation cachée

    Retour :
    - y_pred : vecteur des classes prédites
    """
    probas = predict_proba_mlp(X, parameters, activation=activation)
    y_pred = np.argmax(probas, axis=1)
    return y_pred


def accuracy(y_true, y_pred):
    """
    Calcule l'accuracy du modèle.

    Paramètres :
    - y_true : vecteur des vrais labels
    - y_pred : vecteur des labels prédits

    Retour :
    - acc : proportion de prédictions correctes
    """
    return np.mean(y_true == y_pred)
