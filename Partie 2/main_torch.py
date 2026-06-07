"""
Partie 2 - Version PyTorch (Option B du sujet, p.18).
Memes archi et logique que main_torch.py de la Partie 3, adaptees a CIFAR-10
(images couleur 32x32, 10 classes).
Chargement via torchvision.datasets.CIFAR10 (auto-download si necessaire).
"""
import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Backend : {device}")


# ----- 1. Chargement de CIFAR-10 via torchvision -----
# Stats CIFAR-10 standard (mean/std par canal RGB)
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

# Sur le train : data augmentation (flip horizontal aleatoire)
train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.ToTensor(),                # uint8 [0,255] -> float32 [0,1] et (H,W,C) -> (C,H,W)
    T.Normalize(MEAN, STD),
])
# Sur le test : pas d'augmentation, juste tensor + normalize
test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cifar_data")

print("Chargement CIFAR-10 via torchvision...")
train_ds = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True,
                                        download=True, transform=train_transform)
test_ds  = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False,
                                        download=True, transform=test_transform)
print(f"Train : {len(train_ds)} images - Test : {len(test_ds)} images")

class_names = ["avion", "automobile", "oiseau", "chat", "cerf",
               "chien", "grenouille", "cheval", "bateau", "camion"]

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=0)


# ----- 2. Architecture CNN (identique au main.py NumPy de la Partie 2) -----
#  Entree     (3, 32, 32)
#  Conv 64    (64, 32, 32)
#  Conv3D 64  (64, 32, 32)
#  MaxPool    (64, 16, 16)
#  Conv3D 64  (64, 16, 16)
#  MaxPool    (64,  8,  8)
#  Conv3D 64  (64,  8,  8)
#  Flatten    (4096,)
#  Dense 10   (10,)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * 8 * 8, 10)     # 64 cartes 8x8 = 4096
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.relu(self.conv4(x))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)   # logits (softmax integre dans CrossEntropyLoss)


modele = CNN().to(device)
print(f"Parametres : {sum(p.numel() for p in modele.parameters()):,}")


# ----- 3. Loss + Optimiseur -----
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(modele.parameters(), lr=1e-3, weight_decay=1e-4)


# ----- 4. Entrainement + early stopping -----
EPOCHS = 30
best_acc = 0.0
best_state = None

for epoch in range(EPOCHS):
    t0 = time.time()
    modele.train()
    loss_tot, correct, total = 0.0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
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
    loss_te, correct_te, total_te = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = modele(xb)
            loss_te += criterion(out, yb).item() * xb.size(0)
            correct_te += (out.argmax(dim=1) == yb).sum().item()
            total_te += xb.size(0)
    test_loss = loss_te / total_te
    test_acc = correct_te / total_te

    msg = (f"Epoch {epoch+1}/{EPOCHS} - {time.time()-t0:.1f}s "
           f"- loss {train_loss:.4f} - acc {train_acc:.4f} "
           f"- val_loss {test_loss:.4f} - val_acc {test_acc:.4f}")
    if test_acc > best_acc:
        best_acc = test_acc
        best_state = {k: v.clone() for k, v in modele.state_dict().items()}
        msg += "  *"
    print(msg)

modele.load_state_dict(best_state)
torch.save(modele.state_dict(), "cnn_torch_cifar.pt")
print(f"\n[Early stopping] meilleur acc test = {best_acc:.4f} - sauvegarde dans cnn_torch_cifar.pt")
print(f"Taux d'erreur final : {(1 - best_acc) * 100:.2f} %")


# ----- 5. Accuracy par classe -----
modele.eval()
correct_cls = np.zeros(10, dtype=np.int64)
total_cls   = np.zeros(10, dtype=np.int64)
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        pred = modele(xb).argmax(dim=1).cpu().numpy()
        true = yb.numpy()
        for c in range(10):
            mask = true == c
            total_cls[c] += int(mask.sum())
            correct_cls[c] += int(((pred == true) & mask).sum())

print("\n=== Accuracy par classe ===")
for c in range(10):
    print(f"  {class_names[c]:<12} : {correct_cls[c]:>4}/{total_cls[c]} "
          f"= {correct_cls[c] / total_cls[c]:.2%}")
