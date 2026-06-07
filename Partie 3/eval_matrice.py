"""
Charge le modele sauvegarde (test2.pkl) et calcule la matrice de confusion
sur le test set, sans refaire l'entrainement.
"""
import os
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count())

import cupy as np
import numpy as np_real
import pandas as pd
from PIL import Image

from modele import Modele
from utils import one_hot_encode


BASE_IMG_DIR = r"C:\Users\binis\Downloads\projet_Maths_pour_ML-master avec la partie 3\projet_Maths_pour_ML-master\photo cancer du sein"
LABEL_MAP_BIN = {"BENIGN": 0, "BENIGN_WITHOUT_CALLBACK": 0, "MALIGNANT": 1}


def charger_donnees_mammographies(base_dir, csv_name, taille=(128, 128)):
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
    X_data, y_data = [], []
    for _, row in df.iterrows():
        pid = str(row["image file path"]).split("/")[0]
        chemin = index_jpeg.get(pid)
        if chemin is None or not os.path.exists(chemin):
            continue
        path = row["pathology"]
        if path not in LABEL_MAP_BIN:
            continue
        img = Image.open(chemin).convert("L").resize(taille)
        X_data.append(np_real.array(img, dtype=np_real.float32) / 255.0)
        y_data.append(LABEL_MAP_BIN[path])

    X = np_real.array(X_data, dtype=np_real.float32).reshape(-1, 1, *taille)
    y = np_real.array(y_data, dtype=np_real.int64)
    print(f"[{csv_name}] images chargees : {len(y_data)}")
    return X, y


# --- Chargement test set + modele ---
print("Chargement du test set...")
X_test, y_test = charger_donnees_mammographies(BASE_IMG_DIR, "mass_case_description_test_set.csv")
Y_test = np.asarray(one_hot_encode(y_test, num_classes=2))
X_test = np.asarray(X_test)

print("Chargement du modele sauvegarde...")
modele = Modele.charger("test2.pkl")


# --- Prediction par batchs (pour eviter OOM) ---
print("Prediction sur le test set...")
N = X_test.shape[0]
batch = 16
preds = []
for d in range(0, N, batch):
    f = min(d + batch, N)
    preds.append(modele.forward(X_test[d:f]))
pred_test = np.concatenate(preds, axis=0)

y_pred = np.argmax(pred_test, axis=1)
y_true = np.argmax(Y_test, axis=1)
try:
    y_pred = np.asnumpy(y_pred); y_true = np.asnumpy(y_true)
except AttributeError:
    pass


# --- Matrice de confusion ---
VN = int(((y_pred == 0) & (y_true == 0)).sum())
FP = int(((y_pred == 1) & (y_true == 0)).sum())
FN = int(((y_pred == 0) & (y_true == 1)).sum())
VP = int(((y_pred == 1) & (y_true == 1)).sum())

acc = (VP + VN) / (VP + VN + FP + FN)
precision = VP / (VP + FP) if (VP + FP) else 0
rappel    = VP / (VP + FN) if (VP + FN) else 0
specif    = VN / (VN + FP) if (VN + FP) else 0
f1        = 2 * precision * rappel / (precision + rappel) if (precision + rappel) else 0

print("\n=== Matrice de confusion (test set) ===")
print(f"                     Predit benin    Predit malin")
print(f"Reel benin (0)       VN = {VN:4d}        FP = {FP:4d}")
print(f"Reel malin (1)       FN = {FN:4d}        VP = {VP:4d}")
print()
print(f"Accuracy                       = {acc:.4f}")
print(f"Precision (malin)              = {precision:.4f}")
print(f"Rappel / Sensibilite (malin)   = {rappel:.4f}")
print(f"Specificite (benin)            = {specif:.4f}")
print(f"F1-score (malin)               = {f1:.4f}")
