import random
import time
import pickle
import numpy as np
import pandas as pd


# =========================
# Sauvegarde / chargement
# =========================

def save_results(filepath, data):
    """
    Sauvegarde les résultats dans un fichier pickle.
    Utile si on lance les random search pendant longtemps.
    """
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_results(filepath):
    """
    Recharge des résultats sauvegardés.
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)


# =========================
# Split train / validation
# =========================

def create_train_valid_split(X_train_flat, Y_train, y_train, valid_size=10000):
    """
    Crée un split train / validation simple.

    On garde les dernières images pour la validation.
    """
    X_train_sub = X_train_flat[:-valid_size]
    Y_train_sub = Y_train[:-valid_size]
    y_train_sub = y_train[:-valid_size]

    X_valid = X_train_flat[-valid_size:]
    Y_valid = Y_train[-valid_size:]
    y_valid = y_train[-valid_size:]

    return X_train_sub, Y_train_sub, y_train_sub, X_valid, Y_valid, y_valid


# =========================
# Affichage des résultats
# =========================

def results_to_dataframe(results):
    """
    Convertit une liste de résultats en DataFrame lisible.
    """
    rows = []

    for result in results:
        row = {
            "model": result["model"],
            "trial": result["trial"],
            "valid_accuracy": result["valid_accuracy"],
            "final_loss": result["final_loss"],
            "time_sec": result["time_sec"],
        }

        for key, value in result["params"].items():
            row[key] = value

        rows.append(row)

    df = pd.DataFrame(rows)

    if len(df) > 0:
        df = df.sort_values(by="valid_accuracy", ascending=False)

    return df


def print_best_result(best_result):
    """
    Affiche clairement le meilleur résultat.
    """
    print("\n==============================")
    print("Meilleur résultat")
    print("==============================")
    print("Modèle :", best_result["model"])
    print("Validation accuracy :", best_result["valid_accuracy"])
    print("Final loss :", best_result["final_loss"])
    print("Temps :", round(best_result["time_sec"], 2), "s")
    print("Paramètres :", best_result["params"])


# =========================
# Random search modèle linéaire
# =========================

def random_search_linear_model(
    X_train,
    Y_train,
    X_valid,
    y_valid,
    train_linear_model,
    predict,
    accuracy,
    n_trials=10,
    learning_rate_choices=(0.5, 0.1, 0.05, 0.01),
    epochs_choices=(100, 200, 300),
    input_dim=784,
    num_classes=10,
    save_path=None,
    seed=42
):
    """
    Random search pour le modèle linéaire.
    """
    random.seed(seed)
    np.random.seed(seed)

    results = []
    best_result = None
    best_accuracy = -1

    for trial in range(n_trials):
        learning_rate = random.choice(learning_rate_choices)
        epochs = random.choice(epochs_choices)

        print(f"\n=== Modèle linéaire - Essai {trial + 1}/{n_trials} ===")
        print("learning_rate =", learning_rate)
        print("epochs =", epochs)

        start_time = time.time()

        A, b, loss_history = train_linear_model(
            X_train,
            Y_train,
            input_dim=input_dim,
            num_classes=num_classes,
            learning_rate=learning_rate,
            epochs=epochs
        )

        y_pred_valid = predict(X_valid, A, b)
        valid_accuracy = accuracy(y_valid, y_pred_valid)

        elapsed_time = time.time() - start_time

        result = {
            "model": "Linear",
            "trial": trial + 1,
            "params": {
                "learning_rate": learning_rate,
                "epochs": epochs
            },
            "valid_accuracy": valid_accuracy,
            "final_loss": loss_history[-1],
            "time_sec": elapsed_time,
            "loss_history": loss_history,
            "best_parameters": {
                "A": A,
                "b": b
            }
        }

        results.append(result)

        print("Validation accuracy :", valid_accuracy)
        print("Final loss :", loss_history[-1])
        print("Temps :", round(elapsed_time, 2), "s")

        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            best_result = result

        if save_path is not None:
            save_results(save_path, {
                "results": results,
                "best_result": best_result
            })

    print_best_result(best_result)

    return best_result, results


# =========================
# Random search MLP 1 couche cachée
# =========================

def random_search_mlp_1_hidden(
    X_train,
    Y_train,
    X_valid,
    y_valid,
    train_mlp_1_hidden,
    predict_1_hidden,
    accuracy,
    n_trials=10,
    hidden_dim_choices=(64, 128, 256),
    learning_rate_choices=(0.1, 0.05, 0.01, 0.005),
    epochs_choices=(50, 100, 200),
    input_dim=784,
    num_classes=10,
    save_path=None,
    seed=42
):
    """
    Random search pour le MLP à une couche cachée.
    """
    random.seed(seed)
    np.random.seed(seed)

    results = []
    best_result = None
    best_accuracy = -1

    for trial in range(n_trials):
        hidden_dim = random.choice(hidden_dim_choices)
        learning_rate = random.choice(learning_rate_choices)
        epochs = random.choice(epochs_choices)

        print(f"\n=== MLP 1 couche cachée - Essai {trial + 1}/{n_trials} ===")
        print("hidden_dim =", hidden_dim)
        print("learning_rate =", learning_rate)
        print("epochs =", epochs)

        start_time = time.time()

        A1, b1, A2, b2, loss_history = train_mlp_1_hidden(
            X_train,
            Y_train,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            learning_rate=learning_rate,
            epochs=epochs
        )

        y_pred_valid = predict_1_hidden(X_valid, A1, b1, A2, b2)
        valid_accuracy = accuracy(y_valid, y_pred_valid)

        elapsed_time = time.time() - start_time

        result = {
            "model": "MLP 1 hidden",
            "trial": trial + 1,
            "params": {
                "hidden_dim": hidden_dim,
                "learning_rate": learning_rate,
                "epochs": epochs
            },
            "valid_accuracy": valid_accuracy,
            "final_loss": loss_history[-1],
            "time_sec": elapsed_time,
            "loss_history": loss_history,
            "best_parameters": {
                "A1": A1,
                "b1": b1,
                "A2": A2,
                "b2": b2
            }
        }

        results.append(result)

        print("Validation accuracy :", valid_accuracy)
        print("Final loss :", loss_history[-1])
        print("Temps :", round(elapsed_time, 2), "s")

        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            best_result = result

        if save_path is not None:
            save_results(save_path, {
                "results": results,
                "best_result": best_result
            })

    print_best_result(best_result)

    return best_result, results


# =========================
# Random search MLP 2 couches cachées
# =========================

def random_search_mlp_2_hidden(
    X_train,
    Y_train,
    X_valid,
    y_valid,
    train_mlp_2_hidden,
    predict_2_hidden,
    accuracy,
    n_trials=10,
    hidden_dim1_choices=(128, 256),
    hidden_dim2_choices=(64, 128),
    learning_rate_choices=(0.05, 0.01, 0.005),
    epochs_choices=(50, 100, 200),
    input_dim=784,
    num_classes=10,
    save_path=None,
    seed=42
):
    """
    Random search pour le MLP à deux couches cachées.
    """
    random.seed(seed)
    np.random.seed(seed)

    results = []
    best_result = None
    best_accuracy = -1

    for trial in range(n_trials):
        hidden_dim1 = random.choice(hidden_dim1_choices)
        hidden_dim2 = random.choice(hidden_dim2_choices)
        learning_rate = random.choice(learning_rate_choices)
        epochs = random.choice(epochs_choices)

        print(f"\n=== MLP 2 couches cachées - Essai {trial + 1}/{n_trials} ===")
        print("hidden_dim1 =", hidden_dim1)
        print("hidden_dim2 =", hidden_dim2)
        print("learning_rate =", learning_rate)
        print("epochs =", epochs)

        start_time = time.time()

        A1, b1, A2, b2, A3, b3, loss_history = train_mlp_2_hidden(
            X_train,
            Y_train,
            input_dim=input_dim,
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            num_classes=num_classes,
            learning_rate=learning_rate,
            epochs=epochs
        )

        y_pred_valid = predict_2_hidden(X_valid, A1, b1, A2, b2, A3, b3)
        valid_accuracy = accuracy(y_valid, y_pred_valid)

        elapsed_time = time.time() - start_time

        result = {
            "model": "MLP 2 hidden",
            "trial": trial + 1,
            "params": {
                "hidden_dim1": hidden_dim1,
                "hidden_dim2": hidden_dim2,
                "learning_rate": learning_rate,
                "epochs": epochs
            },
            "valid_accuracy": valid_accuracy,
            "final_loss": loss_history[-1],
            "time_sec": elapsed_time,
            "loss_history": loss_history,
            "best_parameters": {
                "A1": A1,
                "b1": b1,
                "A2": A2,
                "b2": b2,
                "A3": A3,
                "b3": b3
            }
        }

        results.append(result)

        print("Validation accuracy :", valid_accuracy)
        print("Final loss :", loss_history[-1])
        print("Temps :", round(elapsed_time, 2), "s")

        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            best_result = result

        if save_path is not None:
            save_results(save_path, {
                "results": results,
                "best_result": best_result
            })

    print_best_result(best_result)

    return best_result, results