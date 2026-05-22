"""
AutoEncoder – 2 voies (voieHaute + voieBasse)

"""
from datetime import datetime
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.insert(0, os.path.abspath('./pydata'))
from pydata.disdrometre import Disdro
from pydata.fichiers_de_gouttes_sph import dbs_load_bin




file_list = ('Temporel_DBS1_20220720.bin0002',
                     'Temporel_DBS1_20221014.bin0006',
                     'Temporel_DBS1_20221220.bin0002',
                     'Temporel_DBS1_20221227.bin0009',
                     'Temporel_DBS1_20240109.bin0002',
                     'Temporel_DBS1_20240823.bin0003',
                     'Temporel_DBS1_20240823.bin0003',
                     'Temporel_DBS1_20240830.bin0004')
#----CONFIG---------------------------------------
FICHIER       = './bin/'+file_list[0]
DATA_FRACTION = 1  

WINDOW_SIZE   = 512
STRIDE        = 256 

LATENT_DIM    = 64      #8,32
BATCH_SIZE    = 32
N_EPOCHS      = 70
LR            = 1e-3
ARCHITECTURE  = '2voies'


# ── Dataset ----------------------------------------
class PrecipDataset(Dataset):
    def __init__(self):
        raw = dbs_load_bin(FICHIER, Disdro('dbs1'))

        vh = raw['voieHaute'].astype(np.float32)
        vb = raw['voieBasse'].astype(np.float32)

        n  = int(len(vh) * DATA_FRACTION)
        vh = vh[:n]
        vb = vb[:n]

        # normalisation z-score par voie
        vh = (vh - vh.mean()) / (vh.std() + 1e-8)
        vb = (vb - vb.mean()) / (vb.std() + 1e-8)

        # self.signal : shape (2, N)
        self.signal    = np.stack([vh, vb], axis=0).astype(np.float32) #////////!!
        self.n_windows = (n - WINDOW_SIZE) // STRIDE + 1
        print(f"Points : {n:,}  |  Fenêtres : {self.n_windows:,}")

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
      s = idx * STRIDE
      window = self.signal[:, s:s + WINDOW_SIZE].copy()  # shape (2, WINDOW_SIZE)

      # Normalisation par fenêtre (par voie)
      mean_w = window.mean(axis=1, keepdims=True)
      std_w = window.std(axis=1, keepdims=True) + 1e-8
      window = (window - mean_w) / std_w

      x = torch.from_numpy(window)
      return x, x



# ── Modèle 2  : avec 3 conv  ────────────────────────────────────────────────
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Entrée : (Batch, 2, 512)
        self.enc_conv = nn.Sequential(
            # Couche 1 : 512 -> 256
            nn.Conv1d(2, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.5),
            
            # Couche 2 : 256 -> 128
            nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.5),
            
            # Couche 3 : 128 -> 64
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.5),
        )
        
        self._L, self._ch = 64, 64
        self.flat_size = self._L * self._ch # 64 * 64 = 4096

        # MLP Encodeur 
        self.enc_mlp = nn.Sequential(
            nn.Linear(self.flat_size, 128),
            nn.LeakyReLU(0.5),
            nn.Linear(128, LATENT_DIM)
        )

        # ── Décodeur ─────────────────────────────────────
        # MLP Décodeur :
        self.dec_mlp = nn.Sequential(
            nn.Linear(LATENT_DIM, 128),
            nn.LeakyReLU(0.0),
            nn.Linear(128, self.flat_size),
            nn.LeakyReLU(0.5)
        )

        self.dec_conv = nn.Sequential(
            # Couche 1 : 64 -> 128
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.5),
            
            # Couche 2 : 128 -> 256
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.5),
            
            # Couche 3 : 256 -> 512 (2 voies)
            nn.ConvTranspose1d(16, 2, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, x):
        h = self.enc_conv(x).view(x.size(0), -1)
        return self.enc_mlp(h)

    def decode(self, z):
        h   = self.dec_mlp(z).view(z.size(0), self._ch, self._L)
        out = self.dec_conv(h)
        return out

    def forward(self, x):
        return self.decode(self.encode(x))


# ── Entraînement ──────────────────────────────────────────-----------------------
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
    optimizer, mode='min', factor=0.5, patience=5, verbose=True )
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

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{N_EPOCHS}  train={train_losses[-1]:.5f}  val={val_losses[-1]:.5f}")

    # ── Évaluation finale sur l'ensemble de TEST ──────────────────
    from sklearn.metrics import r2_score # pour mesurer la qualité de la reconstruction
    model.eval()
    test_loss = 0.0
    all_x, all_x_hat = [], [] # Pour stocker les signaux et calculer le R2

    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            x_hat = model(x)
            test_loss += criterion(x_hat, x).item() * x.size(0)
            all_x.append(x.cpu().reshape(-1).numpy())
            all_x_hat.append(x_hat.cpu().reshape(-1).numpy())

    test_loss /= n_te

    y_true = np.concatenate(all_x)
    y_pred = np.concatenate(all_x_hat)
    accuracy_r2 = r2_score(y_true, y_pred)

    print("\n" + "═"*50)
    print(f"ERREUR FINALE (MSE) : {test_loss:.5f}")
    print(f"ACCURACY (R² Score) : {accuracy_r2:.2%}")
    print("═"*50 + "\n")

    # PCA sur latents
    from sklearn.decomposition import PCA
    model.eval()
    latents = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            z = model.encode(x)
            latents.append(z.cpu().numpy())
    latents = np.concatenate(latents, axis=0)

    pca = PCA(n_components=2)
    latents_pca = pca.fit_transform(latents)

    plt.figure(figsize=(8, 6))
    plt.scatter(latents_pca[:, 0], latents_pca[:, 1], s=1, alpha=0.5)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.title(f'ACP de l\'espace latent (dim={LATENT_DIM})')
    plt.grid(alpha=0.3)
    plt.show()

    # ── Courbes de loss ───────────────────────────────────────────
    plt.figure(figsize=(8, 3))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses,   label='Val', linestyle='--')
    plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.legend(); plt.grid(alpha=0.3)
    plt.title(f'Loss  latent={LATENT_DIM}')
    plt.tight_layout(); plt.show()

    # ── Reconstruction ────────────────────────────────────
    model.eval()
    with torch.no_grad():
        x_batch, _ = next(iter(test_loader))
        x_hat = model(x_batch.to(device)).cpu().numpy()
    x_np = x_batch.numpy()

    t = np.arange(WINDOW_SIZE) * 8e-5 * 1000   # ms
    # Afficher 3 exemples, chaque exemple a deux sous-graphes (Haute / Basse)
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    for i in range(3):
        # Haute (colonne 0)
        ax = axes[i, 0]
        ax.plot(t, x_np[i, 0],  label='Original',    lw=0.9)
        ax.plot(t, x_hat[i, 0], label='Reconstruit', lw=0.9, linestyle='--')
        ax.set_title(f'Haute - Exemple {i+1}')
        ax.set_xlabel('ms'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # Basse (colonne 1)
        ax = axes[i, 1]
        ax.plot(t, x_np[i, 1],  label='Original',    lw=0.9)
        ax.plot(t, x_hat[i, 1], label='Reconstruit', lw=0.9, linestyle='--')
        ax.set_title(f'Basse - Exemple {i+1}')
        ax.set_xlabel('ms'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle(f'2 voies – Reconstruction  latent={LATENT_DIM}')
    plt.tight_layout(); plt.show()

    print(f"Hyperparamètres : LR={LR}, latent_dim={LATENT_DIM}, batch_size={BATCH_SIZE}, n_epochs={N_EPOCHS}, data_fraction={DATA_FRACTION}, window_size={WINDOW_SIZE}, stride={STRIDE}")
    x_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #torch.save(model.state_dict(), "best_AE_1805_2V_"+x_time +".pth") 
    # "best_AE_1805_2V_2026-05-19_15-43-24.pth"
    
    import seaborn as sns
    import pandas as pd

    
  
    #------- ANALYSE DE L'ESPACE LATENT SUR LES 3 ENSEMBLES -----------------------------------------

    def extract_latent_space(loader):
        """Extrait les vecteurs Z pour un DataLoader donné."""
        all_z = []
        with torch.no_grad():
            for x_batch, _ in loader:
                z = model.encode(x_batch.to(device))
                all_z.append(z.cpu().numpy())
        return np.concatenate(all_z, axis=0)

    # Dictionnaire de DataLoaders 
    loaders = {
        "Entrainement": train_loader,
        "Validation": val_loader,
        "Test": test_loader
    }

    max_dims_pairplot = min(8, LATENT_DIM) # Limite pour éviter de surcharger le graphique

    for nom_ensemble, loader in loaders.items():
        print(f"\nExtraction et Analyse pour l'ensemble : {nom_ensemble}...")
        Z = extract_latent_space(loader)
        
        print(f"  Shape : {Z.shape}")
        print(f"  min={Z.min():.3f}  max={Z.max():.3f}  std={Z.std():.3f}")

        # Conversion en DataFrame Pandas pour faciliter l'analyse
        colonnes = [f'z{i}' for i in range(LATENT_DIM)]
        df_Z = pd.DataFrame(Z, columns=colonnes)


      # --- 1. Matrice de Corrélation ---
        corr_matrix = df_Z.corr()
        
        plt.figure(figsize=(10, 8))
        # CHANGEMENT ICI : ajout de vmin=-1 et vmax=1, et utilisation de 'coolwarm'
        sns.heatmap(corr_matrix, annot=(LATENT_DIM <= 16), cmap='coolwarm', 
                    vmin=-1, vmax=1, center=0, fmt=".2f", square=True, 
                    cbar_kws={'label': 'Coefficient de Pearson'})
        
        plt.title(f"Matrice de Corrélation Linéaire - Ensemble de {nom_ensemble}\n(Archi: {ARCHITECTURE} | Latent: {LATENT_DIM})", fontsize=12)
        plt.tight_layout()
        plt.show()

        # --- 2. Histograms et Histogrammes 2D (Pairplot) ---
        colonnes_pairplot = colonnes[:max_dims_pairplot]
        
        print(f"  Génération du Pairplot de densité pour les {max_dims_pairplot} premières dimensions...")
        
        fig_pair = sns.pairplot(df_Z[colonnes_pairplot], corner=True, kind='hist',
                                plot_kws={'cmap': 'Blues', 'bins': 30})
        
        fig_pair.fig.suptitle(f'Distributions et Densité - Ensemble de {nom_ensemble}', y=1.02, fontsize=12)
        plt.show()
        # gausien + decorrelation => independance


if __name__ == '__main__':
    train()
