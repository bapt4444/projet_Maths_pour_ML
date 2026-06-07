import os
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count())

import cupy as np
np.set_printoptions(suppress=True)

import numpy as np_real
import pandas as pd
from PIL import Image


from modele import Modele
from activation import ReLU, Softmax
from utils import one_hot_encode


BASE_IMG_DIR = r"C:\Users\binis\Downloads\projet_Maths_pour_ML-master avec la partie 3\projet_Maths_pour_ML-master\photo cancer du sein"

LABEL_MAP_BIN = {"BENIGN": 0, "BENIGN_WITHOUT_CALLBACK": 0, "MALIGNANT": 1}


def charger_donnees_mammographies(base_dir, csv_name, taille=(128, 128)):
    """
    Lit le CSV mass_case et matche chaque ligne avec son JPEG via dicom_info.csv.
    Retour : (X, y) numpy ; X shape (N,1,H,W), y ∈ {0,1}.
    """
    # Index PatientID -> chemin jpeg (full mammogram images uniquement)
    df_dicom = pd.read_csv(os.path.join(base_dir, "csv", "dicom_info.csv"))
    df_dicom = df_dicom[df_dicom["SeriesDescription"] == "full mammogram images"]
    df_dicom = df_dicom.dropna(subset=["PatientID", "image_path"])
    index_jpeg = {}
    for pid, rel in zip(df_dicom["PatientID"], df_dicom["image_path"]):
        rel = str(rel).replace("\\", "/")
        if rel.startswith("CBIS-DDSM/"):
            rel = rel[len("CBIS-DDSM/"):]
        index_jpeg.setdefault(pid, os.path.join(base_dir, rel.replace("/", os.sep)))

    df = pd.read_csv(os.path.join(base_dir, "csv", csv_name))
    X_data, y_data, missing = [], [], []
    for _, row in df.iterrows():
        pid = str(row["image file path"]).split("/")[0]
        chemin = index_jpeg.get(pid)
        if chemin is None or not os.path.exists(chemin):
            missing.append(chemin if chemin else pid)
            continue
        path = row["pathology"]
        if path not in LABEL_MAP_BIN:
            continue
        img = Image.open(chemin).convert("L").resize(taille)
        X_data.append(np_real.array(img, dtype=np_real.float32) / 255.0)
        y_data.append(LABEL_MAP_BIN[path])

    X = np_real.array(X_data, dtype=np_real.float32).reshape(-1, 1, *taille)
    y = np_real.array(y_data, dtype=np_real.int64)
    print(f"[{csv_name}] CSV={len(df)}  trouvees={len(y_data)}  manquantes={len(missing)}")
    print(f"  distribution : benin(0)={int((y==0).sum())}  malin(1)={int((y==1).sum())}")
    return X, y


# Charger train et test
X_train, y_train = charger_donnees_mammographies(BASE_IMG_DIR, "mass_case_description_train_set.csv")
X_test, y_test = charger_donnees_mammographies(BASE_IMG_DIR, "mass_case_description_test_set.csv")


# Data augmentation : flips horizontaux ET verticaux (x4)
print("Data augmentation : flips horizontaux + verticaux...")
X_hflip = X_train[:, :, :, ::-1].copy()
X_vflip = X_train[:, :, ::-1, :].copy()
X_hvflip = X_train[:, :, ::-1, ::-1].copy()
X_train = np_real.concatenate([X_train, X_hflip, X_vflip, X_hvflip], axis=0)
y_train = np_real.concatenate([y_train, y_train, y_train, y_train], axis=0)
print(f"Train apres augmentation : {len(y_train)} images")

Y_train = one_hot_encode(y_train, num_classes=2)
Y_test = one_hot_encode(y_test, num_classes=2)
Y_train = np.asarray(Y_train)
Y_test = np.asarray(Y_test)
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)

# Poids de classe : on penalise plus fort les FN (cancer rate)
# poids = 1.0 pour benin, 2.0 pour malin
class_weights = np.asarray([1.0, 3.0], dtype=np.float32)

print(f"X_train : {X_train.shape} - X_test : {X_test.shape}")
print(f"Poids de classe : benin={float(class_weights[0])}  malin={float(class_weights[1])}")



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
#  Dense 2    (2,)  +  Softmax


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
modele.ajouter_couche_dense(Softmax, 2)


print(modele.historique)
modele.entrainer(X_train, Y_train,
                 epochs=60,
                 learning_rate=0.02,
                 batch_size=16,
                 X_test=X_test, Y_test=Y_test,
                 shuffle=True,
                 verbose=True,
                 class_weights=class_weights,
                 early_stopping=True)


modele.sauvegarder(chemin_fichier="test2.pkl")
loss_test, acc_test = modele.evaluer(X_test, Y_test, batch_size=128)
print(f"\nResultats finaux sur le test set :")
print(f"  loss = {loss_test:.4f}")
print(f"  acc  = {acc_test:.4f}")
print(f"  taux d'erreur = {(1-acc_test)*100:.2f} %")


# --- Matrice de confusion sur le test set ---
pred_test = modele.forward(X_test)
y_pred = np.argmax(pred_test, axis=1)
y_true = np.argmax(Y_test, axis=1)
try:
    y_pred = np.asnumpy(y_pred); y_true = np.asnumpy(y_true)
except AttributeError:
    pass

VN = int(((y_pred == 0) & (y_true == 0)).sum())
FP = int(((y_pred == 1) & (y_true == 0)).sum())
FN = int(((y_pred == 0) & (y_true == 1)).sum())
VP = int(((y_pred == 1) & (y_true == 1)).sum())

print("\n--- Matrice de confusion (test set) ---")
print(f"                   Predit benin    Predit malin")
print(f"Reel benin (0)     VN = {VN:4d}      FP = {FP:4d}")
print(f"Reel malin (1)     FN = {FN:4d}      VP = {VP:4d}")

precision = VP / (VP + FP) if (VP + FP) else 0
rappel    = VP / (VP + FN) if (VP + FN) else 0
specif    = VN / (VN + FP) if (VN + FP) else 0
print(f"\nPrecision (malin)          = {precision:.3f}")
print(f"Rappel/Sensibilite (malin) = {rappel:.3f}")
print(f"Specificite (benin)        = {specif:.3f}")
