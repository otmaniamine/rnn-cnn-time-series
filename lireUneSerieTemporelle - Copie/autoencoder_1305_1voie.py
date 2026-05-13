"""
AutoEncoder – voieBasse  (1 seul canal)
le 13/05/2026
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.insert(0, os.path.abspath('./pydata'))
from pydata.disdrometre import Disdro
from pydata.fichiers_de_gouttes_sph import dbs_load_bin

# ══════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════
FICHIER       = './bin/Temporel_DBS1_20220720.bin0002'
DATA_FRACTION = 1  

WINDOW_SIZE   = 512
STRIDE        = 256 

LATENT_DIM    = 32      #8,32
BATCH_SIZE    = 32
N_EPOCHS      = 70  #+ early stopping  !!
LR            = 1e-3     




# ── Dataset ───────────────────────────────────────────────----------------------------------------
class PrecipDataset(Dataset):
    def __init__(self):
        raw = dbs_load_bin(FICHIER, Disdro('dbs1'))
        vb  = raw['voieBasse'].astype(np.float32)

        n  = int(len(vb) * DATA_FRACTION)
        vb = vb[:n]

        # normalisation z-score
        vb = (vb - vb.mean()) / (vb.std() + 1e-8)

        #  self.signal : np.ndarray 
        self.signal    = np.asarray(vb, dtype=np.float32)
        self.n_windows = (n - WINDOW_SIZE) // STRIDE + 1
        print(f"Points : {n:,}  |  Fenêtres : {self.n_windows:,}")

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        s = idx * STRIDE
        x = torch.from_numpy(self.signal[s:s + WINDOW_SIZE])
        x = x.unsqueeze(0)   # (1, 512) ← 1 canal        !!!!!!!!!   
                              #torch.unsqueeze(input, dim) → Tensor
                              #Returns a new tensor with a dimension of size one inserted at the specified position.
        return x, x  #la sortie est la même que l'entrée 





# ── Modèle 2  : avec 3 conv  ────────────────────────────────────────────────--------------------------
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        #  réduction par 2 : k=4, s=2, p=1
        # Entrée : (Batch, 1, 512)
        
        # ── Encodeur ─────────────────────────────── 
        self.enc_conv = nn.Sequential(
            # Couche 1 : 512 -> 256
            nn.Conv1d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.5),
            #nn.Dropout1d(0.1),
            
            # Couche 2 : 256 -> 128
            nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.5),
            #nn.Dropout1d(0.1),
            
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
            nn.LeakyReLU(0.0),
            nn.Linear(128, LATENT_DIM)
        )

        # ── Décodeur ─────────────────────────────────────
        # MLP Décodeur :
        self.dec_mlp = nn.Sequential(
            nn.Linear(LATENT_DIM, 128),
            nn.LeakyReLU(0.0),
            nn.Linear(128, self.flat_size),
            nn.LeakyReLU(0.0)
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
            
            # Couche 3 : 256 -> 512 (Pas d'activation à la fin car Z-score !)
            nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1),
            #nn.Tanh()
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
    from sklearn.metrics import r2_score
    model.eval()
    test_loss = 0.0
    all_x, all_x_hat = [], [] # Pour stocker les signaux et calculer le R2

    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            x_hat = model(x)
            test_loss += criterion(x_hat, x).item() * x.size(0)
            
            all_x.append(x.cpu().view(-1).numpy())
            all_x_hat.append(x_hat.cpu().view(-1).numpy())
    
    test_loss /= n_te
    
    # Calcul du R2 (équivalent de l'accuracy)
    y_true = np.concatenate(all_x)
    y_pred = np.concatenate(all_x_hat)
    accuracy_r2 = r2_score(y_true, y_pred)

    print("\n" + "═"*50)
    print(f"ERREUR FINALE (MSE) : {test_loss:.5f}")
    print(f"ACCURACY (R² Score) : {accuracy_r2:.2%}") # Affiche en %
    print("═"*50 + "\n")
  
    
#if False :   
    #_________pca____________________________________________________________________
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Après l'entraînement, récupérer les latents sur le test set
    model.eval()
    latents = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            z = model.encode(x)  # shape (batch, LATENT_DIM)
            latents.append(z.cpu().numpy())
    latents = np.concatenate(latents, axis=0)
    
   
    pca = PCA(n_components=2)
    latents_pca = pca.fit_transform(latents)
    
    # Visualisation
    plt.figure(figsize=(8, 6))
    plt.scatter(latents_pca[:, 0], latents_pca[:, 1], s=1, alpha=0.5)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.title(f'ACP de l\'espace latent (dim={LATENT_DIM})')
    plt.grid(alpha=0.3)
    plt.show()   
    #_____________________________________________________________________________
    
    
    
    
    

    # ── Courbes ───────────────────────────────────────────
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
    fig, axes = plt.subplots(1, 3, figsize=(14, 3))
    for i, ax in enumerate(axes):
        ax.plot(t, x_np[i, 0],  label='Original',    lw=0.9)
        ax.plot(t, x_hat[i, 0], label='Reconstruit', lw=0.9, linestyle='--')
        ax.set_title(f'Exemple {i+1}')
        ax.set_xlabel('ms'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.suptitle(f'voieBasse – Reconstruction  latent={LATENT_DIM}')
    plt.tight_layout(); plt.show()
    
    print(f"Hyperparamètres : LR={LR}, latent_dim={LATENT_DIM}, batch_size={BATCH_SIZE}, n_epochs={N_EPOCHS}, data_fraction={DATA_FRACTION}, window_size={WINDOW_SIZE}, stride={STRIDE}")





if __name__ == '__main__':
     
    train()
 
    
 
    
 
    
 #---------------output------------------------------------------------------------------------------------------
    
    """ Points : 11,814,912  |  Fenêtres : 23,076
    Paramètres : 1,082,241
    Epoch   1/70  train=0.09818  val=0.03314
    Epoch   5/70  train=0.02519  val=0.01599
    Epoch  10/70  train=0.01620  val=0.03070
    Epoch  15/70  train=0.01481  val=0.01584
    Epoch  20/70  train=0.01207  val=0.01550
    Epoch  25/70  train=0.01000  val=0.00889
    Epoch  30/70  train=0.00686  val=0.00736
    Epoch  35/70  train=0.00624  val=0.00668
    Epoch  40/70  train=0.00464  val=0.00578
    Epoch  45/70  train=0.00475  val=0.00796
    Epoch  50/70  train=0.00438  val=0.00515
    Epoch  55/70  train=0.00415  val=0.00743
    Epoch  60/70  train=0.00388  val=0.00502
    Epoch  65/70  train=0.00306  val=0.00654
    Epoch  70/70  train=0.00421  val=0.00454

    ══════════════════════════════════════════════════
    ERREUR FINALE SUR L'ENSEMBLE DE TEST : 0.00306
    ══════════════════════════════════════════════════

    Hyperparamètres : LR=0.001, latent_dim=32, batch_size=32, n_epochs=70, data_fraction=1, window_size=512, stride=512
    #avec nn.LeakyReLU(0.2)
    
    """
    
    """
    Paramètres : 1,082,241
    Epoch   1/70  train=0.07058  val=0.01693
    Epoch   5/70  train=0.02052  val=0.01208
    Epoch  10/70  train=0.01311  val=0.01508
    Epoch  15/70  train=0.01004  val=0.00785
    Epoch  20/70  train=0.00685  val=0.00795
    Epoch  25/70  train=0.00523  val=0.00759
    Epoch  30/70  train=0.00406  val=0.00801
    Epoch  35/70  train=0.00387  val=0.00336
    Epoch  40/70  train=0.00378  val=0.00438
    Epoch  45/70  train=0.00332  val=0.00423
    Epoch  50/70  train=0.00279  val=0.00427
    Epoch  55/70  train=0.00248  val=0.00446
    Epoch  60/70  train=0.00260  val=0.00325
    Epoch  65/70  train=0.00225  val=0.00291
    Epoch  70/70  train=0.00225  val=0.00330

    ══════════════════════════════════════════════════
    ERREUR FINALE SUR L'ENSEMBLE DE TEST : 0.00339
    ══════════════════════════════════════════════════

    Hyperparamètres : LR=0.001, latent_dim=32, batch_size=32, n_epochs=70, data_fraction=1, window_size=512, stride=256
    
    #avec nn.LeakyReLU(0.2)
    """
    
    """Points : 11,814,912  |  Fenêtres : 46,151
    Paramètres : 1,082,241
    Epoch   1/70  train=0.08031  val=0.01739
    Epoch   5/70  train=0.01933  val=0.02452
    Epoch  10/70  train=0.01442  val=0.01899
    Epoch  15/70  train=0.01231  val=0.01146
    Epoch  20/70  train=0.00944  val=0.00831
    Epoch  25/70  train=0.00734  val=0.00668
    Epoch  30/70  train=0.00590  val=0.00735
    Epoch  35/70  train=0.00459  val=0.00410
    Epoch  40/70  train=0.00394  val=0.00311
    Epoch  45/70  train=0.00342  val=0.00346
    Epoch  50/70  train=0.00336  val=0.00329
    Epoch  55/70  train=0.00302  val=0.00239
    Epoch  60/70  train=0.00291  val=0.00656
    Epoch  65/70  train=0.00243  val=0.00350
    Epoch  70/70  train=0.00265  val=0.00269

    ══════════════════════════════════════════════════
    ERREUR FINALE SUR L'ENSEMBLE DE TEST : 0.00287 ********************** 
    ══════════════════════════════════════════════════

    Hyperparamètres : LR=0.001, latent_dim=32, batch_size=32, n_epochs=70, data_fraction=1, window_size=512, stride=256
    #avec nn.LeakyReLU(0.5)
    """
    #torch.save(model.state_dict(), "best_autoencoder.pth")