"""
AutoEncoder – voieBasse + voieHaute (2 canaux)
===============================================
Basé sur le meilleur code 1 voie.

3 changements par rapport à la version 1 voie :
  1. Dataset   : charge les 2 voies → tenseur (2, 512) au lieu de (1, 512)
  2. Modèle    : Conv1d(1, ...) → Conv1d(2, ...)  et  ConvTranspose1d(..., 1) → (..., 2)
  3. Visualisation : 2 lignes par exemple (voieBasse + voieHaute)

Stratégie pour le délai variable entre les 2 voies :
  → on ne le calcule pas manuellement
  → on donne les 2 voies ensemble comme 2 canaux
  → le réseau apprend lui-même la relation temporelle entre elles
"""

from datetime import datetime
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.abspath('./pydata'))
from pydata.disdrometre import Disdro
from pydata.fichiers_de_gouttes_sph import dbs_load_bin

# ══════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════
file_list = ('Temporel_DBS1_20220720.bin0002',
             'Temporel_DBS1_20221014.bin0006',
             'Temporel_DBS1_20221220.bin0002',
             'Temporel_DBS1_20221227.bin0009',
             'Temporel_DBS1_20240109.bin0002',
             'Temporel_DBS1_20240823.bin0003',
             'Temporel_DBS1_20240830.bin0004')

FICHIER     = './bin/' + file_list[0]
DATA_FRACTION = 0.00001

WINDOW_SIZE = 512
STRIDE      = 256

LATENT_DIM  = 32
BATCH_SIZE  = 32
N_EPOCHS    = 50
LR          = 1e-3
# ══════════════════════════════════════════════════════════


# ── CHANGEMENT 1 : Dataset 2 voies ───────────────────────
class PrecipDataset(Dataset):
    def __init__(self):
        raw = dbs_load_bin(FICHIER, Disdro('dbs1'))

        vb = raw['voieBasse'].astype(np.float32)
        vh = raw['voieHaute'].astype(np.float32)   # ← ajout

        n  = int(len(vb) * DATA_FRACTION)
        vb, vh = vb[:n], vh[:n]

        # normalisation z-score globale (voie par voie)
        vb = (vb - vb.mean()) / (vb.std() + 1e-8)
        vh = (vh - vh.mean()) / (vh.std() + 1e-8)   # ← ajout

        # shape (2, N) : 2 voies concaténées sur l'axe 0
        self.signal = np.stack([vb, vh], axis=0).astype(np.float32)   # ← ajout
        self.n_windows = (n - WINDOW_SIZE) // STRIDE + 1
        print(f"Points : {n:,}  |  Fenêtres : {self.n_windows:,}  |  2 voies")

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        s = idx * STRIDE
        window = self.signal[:, s:s + WINDOW_SIZE].copy()   # (2, 512)

        # normalisation par fenêtre (voie par voie)
        for c in range(2):
            mean_w = window[c].mean()
            std_w  = window[c].std() + 1e-8
            window[c] = (window[c] - mean_w) / std_w

        x = torch.from_numpy(window)   # (2, 512) — pas besoin de unsqueeze
        return x, x


# ── CHANGEMENT 2 : Modèle 2 canaux ───────────────────────
class AutoEncoder(nn.Module):
    """
    Identique à la version 1 voie, avec 2 modifications :
      - Conv1d(1, 16, ...) → Conv1d(2, 16, ...)   (entrée 2 canaux)
      - ConvTranspose1d(16, 1, ...) → (..., 2)     (sortie 2 canaux)
    """
    def __init__(self):
        super().__init__()

        # ── Encodeur Conv ─────────────────────────────────
        self.enc_conv = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=4, stride=2, padding=1),   # ← 1→2
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.5),

            nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.5),

            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.5),
        )

        self._L, self._ch = 64, 64
        self.flat_size = self._L * self._ch   # 4096

        # ── Encodeur MLP ──────────────────────────────────
        self.enc_mlp = nn.Sequential(
            nn.Linear(self.flat_size, 128),
            nn.LeakyReLU(0.5),
            nn.Linear(128, LATENT_DIM),
        )

        # ── Décodeur MLP ──────────────────────────────────
        self.dec_mlp = nn.Sequential(
            nn.Linear(LATENT_DIM, 128),
            nn.LeakyReLU(0.0),
            nn.Linear(128, self.flat_size),
            nn.LeakyReLU(0.5),
        )

        # ── Décodeur Conv ─────────────────────────────────
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.5),

            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.5),

            nn.ConvTranspose1d(16, 2, kernel_size=4, stride=2, padding=1),   # ← 1→2
        )

    def encode(self, x):
        h = self.enc_conv(x).view(x.size(0), -1)
        return self.enc_mlp(h)

    def decode(self, z):
        h = self.dec_mlp(z).view(z.size(0), self._ch, self._L)
        return self.dec_conv(h)

    def forward(self, x):
        return self.decode(self.encode(x))


# ── Entraînement ──────────────────────────────────────────
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    ds   = PrecipDataset()
    n    = len(ds)
    n_tr = int(0.70 * n); n_val = int(0.15 * n); n_te = n - n_tr - n_val
    tr, val, te = random_split(ds, [n_tr, n_val, n_te],
                               generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(tr,  batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(te,  batch_size=BATCH_SIZE, shuffle=False)

    model     = AutoEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.MSELoss()
    print(f"Paramètres : {sum(p.numel() for p in model.parameters()):,}")

    train_losses, val_losses = [], []

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        tl = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), x)
            loss.backward()
            optimizer.step()
            tl += loss.item() * x.size(0)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                vl += criterion(model(x), x).item() * x.size(0)

        train_losses.append(tl / n_tr)
        val_losses.append(vl / n_val)
        scheduler.step(val_losses[-1])

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{N_EPOCHS}  train={train_losses[-1]:.5f}  val={val_losses[-1]:.5f}")

    # ── Test final + R² ───────────────────────────────────
    model.eval()
    test_loss = 0.0
    all_x, all_x_hat = [], []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            x_hat = model(x)
            test_loss += criterion(x_hat, x).item() * x.size(0)
            all_x.append(x.cpu().view(-1).numpy())
            all_x_hat.append(x_hat.cpu().view(-1).numpy())

    test_loss /= n_te
    r2 = r2_score(np.concatenate(all_x), np.concatenate(all_x_hat))
    print(f"\n{'═'*50}")
    print(f"ERREUR FINALE (MSE) : {test_loss:.5f}")
    print(f"ACCURACY (R²)       : {r2:.2%}")
    print(f"{'═'*50}\n")

    # ── PCA de l'espace latent ────────────────────────────
    latents = []
    with torch.no_grad():
        for x, _ in test_loader:
            latents.append(model.encode(x.to(device)).cpu().numpy())
    Z = np.concatenate(latents, axis=0)
    print(f"Espace latent : {Z.shape}  min={Z.min():.3f}  max={Z.max():.3f}")

    pca = PCA(n_components=2)
    Z_pca = pca.fit_transform(Z)
    plt.figure(figsize=(7, 5))
    plt.scatter(Z_pca[:, 0], Z_pca[:, 1], s=1, alpha=0.4)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.title(f'ACP espace latent (dim={LATENT_DIM}) — 2 voies')
    plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

    # ── Courbes de perte ──────────────────────────────────
    plt.figure(figsize=(8, 3))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses,   label='Val', linestyle='--')
    plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.legend(); plt.grid(alpha=0.3)
    plt.title(f'Loss  latent={LATENT_DIM}  — 2 voies')
    plt.tight_layout(); plt.show()

    # ── CHANGEMENT 3 : Reconstruction 2 voies ─────────────
    model.eval()
    with torch.no_grad():
        x_batch, _ = next(iter(test_loader))
        x_hat = model(x_batch.to(device)).cpu().numpy()
    x_np = x_batch.numpy()   # (B, 2, 512)

    t = np.arange(WINDOW_SIZE) * 8e-5 * 1000   # ms
    noms = ['voieBasse', 'voieHaute']
    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    for j in range(2):           # voie
        for i in range(3):       # exemple
            ax = axes[j, i]
            ax.plot(t, x_np[i, j],  label='Original',    lw=0.9)
            ax.plot(t, x_hat[i, j], label='Reconstruit', lw=0.9, linestyle='--')
            ax.set_title(f'{noms[j]} – ex {i+1}')
            ax.set_xlabel('ms'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.suptitle(f'Reconstruction 2 voies  latent={LATENT_DIM}')
    plt.tight_layout(); plt.show()

    # ── Histogrammes espace latent ────────────────────────
    n_cols = min(8, LATENT_DIM)
    n_rows = (LATENT_DIM + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5*n_cols, 2.5*n_rows))
    axes = np.array(axes).flatten()
    for dim in range(LATENT_DIM):
        axes[dim].hist(Z[:, dim], bins=30, color='steelblue', alpha=0.7)
        axes[dim].set_title(f'z[{dim}]', fontsize=9)
        axes[dim].tick_params(labelsize=7)
    for ax in axes[LATENT_DIM:]:
        ax.set_visible(False)
    plt.suptitle(f'Distribution latente (dim={LATENT_DIM}) — 2 voies')
    plt.tight_layout(); plt.show()

    print(f"Hyperparamètres : LR={LR}, latent={LATENT_DIM}, batch={BATCH_SIZE}, "
          f"epochs={N_EPOCHS}, fraction={DATA_FRACTION}, window={WINDOW_SIZE}, stride={STRIDE}")

    x_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    #torch.save(model.state_dict(), f"best_AE_2voies_{x_time}.pth")
    #sauvgarde dans un dossier "models" ?
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/best_AE_2voies_{x_time}.pth")


if __name__ == '__main__':
    train()