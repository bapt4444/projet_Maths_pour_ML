"""
Balaye plusieurs seuils de decision sur la sortie softmax du modele sauvegarde,
afin d'illustrer le compromis sensibilite / specificite.
Aucun reentrainement : on reutilise test2.pkl.
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
X_test = np.asarray(X_test)
y_true = y_test  # deja un numpy int64

print("Chargement du modele...")
modele = Modele.charger("test2.pkl")


# --- Prediction (probabilites softmax) ---
print("Calcul des probabilites...")
N = X_test.shape[0]
batch = 16
preds = []
for d in range(0, N, batch):
    f = min(d + batch, N)
    preds.append(modele.forward(X_test[d:f]))
proba = np.concatenate(preds, axis=0)         # forme (N, 2)
try:
    proba = np.asnumpy(proba)
except AttributeError:
    pass
p_malin = proba[:, 1]                          # P(malin)


# --- Balayage des seuils ---
print()
print("=== Sensibilite / specificite selon le seuil de decision ===")
print(f"{'Seuil':>6} | {'VN':>4} {'FP':>4} {'FN':>4} {'VP':>4} | "
      f"{'Sensib.':>8} {'Specif.':>8} {'Precis.':>8} {'Acc':>7} {'F1':>6}")
print("-" * 80)

for seuil in [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10]:
    y_pred = (p_malin >= seuil).astype(int)

    VN = int(((y_pred == 0) & (y_true == 0)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    VP = int(((y_pred == 1) & (y_true == 1)).sum())

    sensib = VP / (VP + FN) if (VP + FN) else 0
    specif = VN / (VN + FP) if (VN + FP) else 0
    precis = VP / (VP + FP) if (VP + FP) else 0
    acc    = (VP + VN) / (VP + VN + FP + FN)
    f1     = 2 * precis * sensib / (precis + sensib) if (precis + sensib) else 0

    print(f"{seuil:>6.2f} | {VN:>4d} {FP:>4d} {FN:>4d} {VP:>4d} | "
          f"{sensib:>7.2%} {specif:>7.2%} {precis:>7.2%} {acc:>6.2%} {f1:>5.2%}")

print()
print("Lecture : pour chaque seuil s, on classe en 'malin' si P(malin) >= s.")
print("Plus s est petit, plus on detecte de cancers (sensibilite up), au prix")
print("de plus de fausses alertes (specificite down).")
