"""
Matrice de confusion sur le TRAIN set (avec le modele PyTorch sauvegarde).
Permet de comparer train vs test et detecter un eventuel sur-apprentissage.
"""
import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn


BASE_IMG_DIR = r"C:\Users\binis\Downloads\projet_Maths_pour_ML-master avec la partie 3\projet_Maths_pour_ML-master\photo cancer du sein"
LABEL_MAP_BIN = {"BENIGN": 0, "BENIGN_WITHOUT_CALLBACK": 0, "MALIGNANT": 1}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Backend : {device}")


# ----- Chargement -----
def charger_donnees(base_dir, csv_name, taille=(224, 224)):
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
        X_data.append(np.array(img, dtype=np.float32) / 255.0)
        y_data.append(LABEL_MAP_BIN[path])

    X = np.array(X_data, dtype=np.float32).reshape(-1, 1, *taille)
    y = np.array(y_data, dtype=np.int64)
    print(f"[{csv_name}] {len(y_data)} images chargees "
          f"(benin = {(y == 0).sum()}, malin = {(y == 1).sum()})")
    return X, y


# ----- Architecture identique a main_torch.py -----
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4   = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn5   = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 28 * 28, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = torch.relu(self.bn5(self.conv5(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


print("Chargement TRAIN set...")
X_train, y_train = charger_donnees(BASE_IMG_DIR, "mass_case_description_train_set.csv")

print("Chargement du modele PyTorch sauvegarde...")
modele = CNN().to(device)
modele.load_state_dict(torch.load("cnn_torch.pt", map_location=device))
modele.eval()


# ----- Prediction par batchs (1318 images en 224x224 sature la VRAM) -----
print("Prediction sur le train set...")
N = X_train.shape[0]
batch = 32
y_pred = np.empty(N, dtype=np.int64)
with torch.no_grad():
    for d in range(0, N, batch):
        f = min(d + batch, N)
        xb = torch.from_numpy(X_train[d:f]).to(device)
        out = modele(xb)
        y_pred[d:f] = out.argmax(dim=1).cpu().numpy()


# ----- Matrice de confusion -----
y_true = y_train
VN = int(((y_pred == 0) & (y_true == 0)).sum())
FP = int(((y_pred == 1) & (y_true == 0)).sum())
FN = int(((y_pred == 0) & (y_true == 1)).sum())
VP = int(((y_pred == 1) & (y_true == 1)).sum())

acc = (VP + VN) / (VP + VN + FP + FN)
precision = VP / (VP + FP) if (VP + FP) else 0
rappel    = VP / (VP + FN) if (VP + FN) else 0
specif    = VN / (VN + FP) if (VN + FP) else 0
f1        = 2 * precision * rappel / (precision + rappel) if (precision + rappel) else 0

print("\n=== Matrice de confusion (TRAIN set) ===")
print(f"                   Predit benin    Predit malin")
print(f"Reel benin (0)     VN = {VN:4d}        FP = {FP:4d}")
print(f"Reel malin (1)     FN = {FN:4d}        VP = {VP:4d}")
print()
print(f"Accuracy                       = {acc:.4f}")
print(f"Precision (malin)              = {precision:.4f}")
print(f"Rappel / Sensibilite (malin)   = {rappel:.4f}")
print(f"Specificite (benin)            = {specif:.4f}")
print(f"F1-score (malin)               = {f1:.4f}")
