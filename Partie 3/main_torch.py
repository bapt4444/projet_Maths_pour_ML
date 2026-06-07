"""
Partie 3 - Version PyTorch (Option B du sujet, p.18).
Architecture identique a notre CNN NumPy + BatchNorm + Adam + Dropout +
data augmentation + early stopping + scheduler.
"""
import os
import time
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import TensorDataset, DataLoader


BASE_IMG_DIR = r"C:\Users\binis\Downloads\projet_Maths_pour_ML-master avec la partie 3\projet_Maths_pour_ML-master\photo cancer du sein"
LABEL_MAP_BIN = {"BENIGN": 0, "BENIGN_WITHOUT_CALLBACK": 0, "MALIGNANT": 1}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Backend : {device}")


# ----- 1. Chargement des donnees -----
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


print("Chargement des donnees...")
X_train, y_train = charger_donnees(BASE_IMG_DIR, "mass_case_description_train_set.csv")
X_test, y_test = charger_donnees(BASE_IMG_DIR, "mass_case_description_test_set.csv")


# ----- 2. Architecture CNN avec BatchNorm -----
class CNN(nn.Module):
    """5 convolutions 3x3 + BatchNorm + 3 max-pool + dense, comme le sujet."""

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
        # apres 3 pools : 224 / 8 = 28 ; 64 cartes 28x28 = 50176
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


modele = CNN().to(device)
print(f"Parametres : {sum(p.numel() for p in modele.parameters()):,}")


# ----- 3. Tenseurs + DataLoader -----
X_train_t = torch.from_numpy(X_train)
y_train_t = torch.from_numpy(y_train)
X_test_t = torch.from_numpy(X_test).to(device)
y_test_t = torch.from_numpy(y_test).to(device)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                          batch_size=16, shuffle=True)


# ----- 4. Loss + Optimiseur + Scheduler -----
# Poids de classe plus modere : on incite a detecter le cancer sans
# trop sur-predire la classe maligne (FP).
class_weights = torch.tensor([1.0, 1.3], device=device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(modele.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                 factor=0.5, patience=3)


# ----- 5. Augmentation enrichie : flips + rotation + luminosite + contraste -----
def augmenter(xb):
    # flip horizontal
    if torch.rand(1).item() < 0.5:
        xb = torch.flip(xb, dims=[3])
    # flip vertical
    if torch.rand(1).item() < 0.5:
        xb = torch.flip(xb, dims=[2])
    # rotation aleatoire +/-20 degres
    angle = (torch.rand(1).item() - 0.5) * 40.0
    xb = TF.rotate(xb, angle)
    # jitter de luminosite : x *= facteur dans [0.8, 1.2]
    bright = 0.8 + 0.4 * torch.rand(1).item()
    xb = (xb * bright).clamp(0.0, 1.0)
    # jitter de contraste : ecart par rapport a la moyenne
    contrast = 0.8 + 0.4 * torch.rand(1).item()
    mean = xb.mean(dim=(2, 3), keepdim=True)
    xb = ((xb - mean) * contrast + mean).clamp(0.0, 1.0)
    return xb


# ----- 6. Entrainement -----
EPOCHS = 40
best_acc = 0.0
best_state = None

for epoch in range(EPOCHS):
    t0 = time.time()
    modele.train()
    loss_tot, correct, total = 0.0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        xb = augmenter(xb)

        optimizer.zero_grad()
        out = modele(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        loss_tot += loss.item() * xb.size(0)
        correct += (out.argmax(dim=1) == yb).sum().item()
        total += xb.size(0)
    train_loss = loss_tot / total
    train_acc = correct / total

    modele.eval()
    with torch.no_grad():
        out_te = modele(X_test_t)
        test_loss = criterion(out_te, y_test_t).item()
        test_acc = (out_te.argmax(dim=1) == y_test_t).float().mean().item()

    scheduler.step(test_acc)
    lr = optimizer.param_groups[0]["lr"]

    msg = (f"Epoch {epoch+1}/{EPOCHS} - {time.time()-t0:.1f}s - lr {lr:.0e} "
           f"- loss {train_loss:.4f} - acc {train_acc:.4f} "
           f"- val_loss {test_loss:.4f} - val_acc {test_acc:.4f}")
    if test_acc > best_acc:
        best_acc = test_acc
        best_state = {k: v.clone() for k, v in modele.state_dict().items()}
        msg += "  *"
    print(msg)

modele.load_state_dict(best_state)
torch.save(modele.state_dict(), "cnn_torch.pt")
print(f"\n[Early stopping] meilleur acc test = {best_acc:.4f} - sauvegarde dans cnn_torch.pt")


# ----- 7. Matrice de confusion -----
modele.eval()
with torch.no_grad():
    out = modele(X_test_t)
    y_pred = out.argmax(dim=1).cpu().numpy()
    y_true = y_test_t.cpu().numpy()

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
print(f"                   Predit benin    Predit malin")
print(f"Reel benin (0)     VN = {VN:4d}        FP = {FP:4d}")
print(f"Reel malin (1)     FN = {FN:4d}        VP = {VP:4d}")
print()
print(f"Accuracy                       = {acc:.4f}")
print(f"Precision (malin)              = {precision:.4f}")
print(f"Rappel / Sensibilite (malin)   = {rappel:.4f}")
print(f"Specificite (benin)            = {specif:.4f}")
print(f"F1-score (malin)               = {f1:.4f}")
