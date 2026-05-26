# -*- coding: utf-8 -*-
"""
Le 25 05 2026
Étape 2 : RNN (LSTM) sur espace latent du DBS1
Fusion et amélioration des approches précédentes

Approche :
1. Extraction des features (z=64 + stats=4) par l'AE pré-entraîné
2. LSTM auto-supervisé : prédit la séquence de latents suivante
3. Extraction physique : détection de vallées + loi de Gunn-Kinzer
4. Visualisations complètes des résultats
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

from autoencoder_1805_2V import AutoEncoder

# =====================================================================
# CONFIGURATION
# =====================================================================
FICHIER_BIN = './bin/Temporel_DBS1_20220720.bin0002'
AE_WEIGHTS  = 'best_AE_1805_2V_2026-05-19_15-43-24.pth' 

WINDOW_SIZE = 512
STRIDE      = 256
LATENT_DIM  = 64
SEQ_LEN     = 40      
BATCH_SIZE  = 64
N_EPOCHS    = 180
LR          = 1e-3
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paramètres extraction physique
DISTANCE_FAISCEAUX_MM = 2.0  # Distance vb/vh = 2mm
TE = 8e-5                   # Période d'échantillonnage (1/12500 Hz)
SEUIL_VALLEE = 0.15         # Seuil relatif pour détection vallée


# =====================================================================
# 1. EXTRACTION DES CARACTÉRISTIQUES (68 dimensions)
# =====================================================================
def extract_latent_features(ae_model, fichier, device):
    """
    Extrait pour chaque fenêtre : [z(64) | mu_vh | sigma_vh | mu_vb | sigma_vb]
    Traitement par batch pour efficacité mémoire.
    """
    print(f"Chargement données brutes : {fichier}")
    raw = dbs_load_bin(fichier, Disdro('dbs1'))
    
    vh = raw['voieHaute'].astype(np.float32)
    vb = raw['voieBasse'].astype(np.float32)
    
    # Normalisation globale
    vh = (vh - vh.mean()) / (vh.std() + 1e-8)
    vb = (vb - vb.mean()) / (vb.std() + 1e-8)
    signal = np.stack([vh, vb], axis=0)  # (2, N)
    
    n_points = signal.shape[1]
    n_windows = (n_points - WINDOW_SIZE) // STRIDE + 1
    
    features = np.zeros((n_windows, LATENT_DIM + 4), dtype=np.float32)
    
    print(f"Extraction de {n_windows:,} fenêtres...")
    ae_model.eval()
    
    batch_size = 512
    for i in range(0, n_windows, batch_size):
        end = min(i + batch_size, n_windows)
        
        windows = []
        stats = []
        for j in range(i, end):
            s = j * STRIDE
            w = signal[:, s:s + WINDOW_SIZE].copy()
            
            mu = w.mean(axis=1, keepdims=True)
            sigma = w.std(axis=1, keepdims=True) + 1e-8
            stats.append([mu[0,0], sigma[0,0], mu[1,0], sigma[1,0]])
            
            w_norm = (w - mu) / sigma
            windows.append(w_norm)
        
        windows_t = torch.tensor(np.array(windows), dtype=torch.float32).to(device)
        
        with torch.no_grad():
            z = ae_model.encode(windows_t).cpu().numpy()
        
        features[i:end, :LATENT_DIM] = z
        features[i:end, LATENT_DIM:] = np.array(stats)
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Batch {i//batch_size + 1}/{(n_windows-1)//batch_size + 1}")
    
    print(f"Features shape : {features.shape}")
    return features


class PrecipSequenceDataset(Dataset):
    """Dataset de séquences temporelles de features."""
    def __init__(self, features, seq_len):
        self.features = features
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.features) - self.seq_len
        
    def __getitem__(self, idx):
        X = self.features[idx : idx + self.seq_len]      # (seq_len, 68)
        Y = self.features[idx + 1 : idx + self.seq_len + 1]  # (seq_len, 68)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)


# =====================================================================
# 2. ARCHITECTURE RNN (LSTM)

class DisdroRNN(nn.Module):
    """
    LSTM avec deux têtes :
    - next_step : prédiction auto-supervisée de la séquence suivante
    - physics : extraction de caractéristiques physiques (D, V, N)
    """
    def __init__(self, input_dim=68, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Tête 1 : Prédiction auto-supervisée (séquence suivante)
        self.fc_next = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Tête 2 : Extraction physique (D, V, N_gouttes)
        self.fc_phys = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout/2),
            nn.Linear(64, 3)  # [D_est, V_est, N_est]
        )
    
    def forward(self, x):
        # x : (batch, seq_len, 68)
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Prédiction pour chaque pas de temps
        next_pred = self.fc_next(lstm_out)  # (batch, seq_len, 68)
        
        # Extraction physique depuis le dernier état caché
        phys = self.fc_phys(lstm_out[:, -1, :])  # (batch, 3)
        
        return next_pred, phys


# =====================================================================
# 3. FONCTIONS D'EXTRACTION PHYSIQUE

def gunn_kinzer(d):
    """Loi de Gunn-Kinzer : vitesse terminale de chute d'une goutte."""
    return 9.40 * (1 - np.exp(-(1.57e3) * (d * 1e-3)**1.15))

def detect_valleys(signal_1d, baseline, threshold_factor=SEUIL_VALLEE):
    """
    Détecte les vallées dans un signal 1D.
    Retourne liste de dict avec start, end, depth, width, min_idx.
    """
    dist = baseline - signal_1d
    threshold = threshold_factor * np.std(signal_1d)
    
    below = dist > threshold
    valleys = []
    i = 0
    while i < len(below):
        if below[i]:
            start = i
            while i < len(below) and below[i]:
                i += 1
            end = i
            
            seg = signal_1d[start:end]
            depth = baseline - np.min(seg)
            width_ms = (end - start) * TE * 1000
            min_idx = start + np.argmin(seg)
            
            valleys.append({
                'start': start, 'end': end,
                'depth': depth, 'width_ms': width_ms,
                'min_idx': min_idx,
                't_start_ms': start * TE * 1000,
                't_end_ms': end * TE * 1000,
                't_min_ms': min_idx * TE * 1000
            })
        else:
            i += 1
    return valleys

def estimate_drops(valleys_vb, valleys_vh, stats_vb, stats_vh):
    """
    Apparie les vallées vb/vh et estime D, V pour chaque goutte.
    stats = [mu, sigma] pour chaque voie.
    """
    props = []
    used_vh = set()
    
    for vb in valleys_vb:
        best_match = None
        best_dt = float('inf')
        
        for i, vh in enumerate(valleys_vh):
            if i in used_vh:
                continue
            dt_idx = abs(vb['min_idx'] - vh['min_idx'])
            dt_s = dt_idx * TE
            if dt_s < best_dt and dt_s < 0.05:  # max 50ms
                best_dt = dt_s
                best_match = i
        
        if best_match is not None:
            vh = valleys_vh[best_match]
            used_vh.add(best_match)
            
            # Vitesse : V = 2mm / delta_t
            dt_s = abs(vb['min_idx'] - vh['min_idx']) * TE
            V = DISTANCE_FAISCEAUX_MM / (dt_s * 1000) if dt_s > 1e-6 else np.nan
            
            # Diamètre : proportionnel à la profondeur moyenne absolue
            depth_vb_abs = vb['depth'] * stats_vb[1]
            depth_vh_abs = vh['depth'] * stats_vh[1]
            D_est = np.mean([depth_vb_abs, depth_vh_abs]) * 0.05  # facteur empirique à calibrer
            
            props.append({
                'D_mm': D_est,
                'V_ms': V,
                't_s': vb['min_idx'] * TE,
                'delta_t_ms': dt_s * 1000,
                'depth_vb': vb['depth'],
                'depth_vh': vh['depth']
            })
    
    return props


# =====================================================================
# 4. ENTRAÎNEMENT

def train_rnn(features):
    dataset = PrecipSequenceDataset(features, SEQ_LEN)
    
    # Split 70/15/15
    n = len(dataset)
    n_tr = int(0.70 * n)
    n_val = int(0.15 * n)
    n_te = n - n_tr - n_val
    
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_tr, n_val, n_te],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = DisdroRNN(input_dim=LATENT_DIM + 4).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    criterion = nn.MSELoss()
    
    train_losses, val_losses = [], []
    best_val = float('inf')
    
    print(f"\nParamètres RNN : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Séquences : train={n_tr:,}, val={n_val:,}, test={n_te:,}")
    print("\n--- Entraînement RNN (Auto-supervisé) ---")
    
    for epoch in range(1, N_EPOCHS + 1):
        # Train
        model.train()
        tl = 0.0
        for X, Y in train_loader:
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            
            optimizer.zero_grad()
            pred, _ = model(X)
            loss = criterion(pred, Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item() * X.size(0)
        
        # Val
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for X, Y in val_loader:
                X, Y = X.to(DEVICE), Y.to(DEVICE)
                pred, _ = model(X)
                vl += criterion(pred, Y).item() * X.size(0)
        
        train_losses.append(tl / n_tr)
        val_losses.append(vl / n_val)
        scheduler.step(val_losses[-1])
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{N_EPOCHS} | "
                  f"Train={train_losses[-1]:.6f} | Val={val_losses[-1]:.6f}")
        
        if val_losses[-1] < best_val:
            best_val = val_losses[-1]
            torch.save(model.state_dict(), 'best_RNN_etape2.pth')
    
    # Test final
    model.load_state_dict(torch.load('best_RNN_etape2.pth', map_location=DEVICE))
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X, Y in test_loader:
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            pred, _ = model(X)
            test_loss += criterion(pred, Y).item() * X.size(0)
    test_loss /= n_te
    
    print(f"\n{'='*50}")
    print(f"TEST MSE : {test_loss:.6f}")
    print(f"Best Val : {best_val:.6f}")
    print(f"{'='*50}")
    
    return model, train_losses, val_losses, test_loss


# =====================================================================
# 5. VISUALISATIONS

def visualize_all(rnn_model, ae_model, features, train_losses, val_losses):
    rnn_model.eval()
    ae_model.eval()
    
    # --- Figure 1 : Courbes de loss ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(train_losses, label='Train', color='steelblue')
    axes[0].plot(val_losses, label='Val', color='coral', linestyle='--')
    axes[0].set_title('Convergence RNN')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # --- Figure 1b : Scatter prédiction vs réalité ---
    # Échantillon de prédictions sur le test set
    idx_test = 8000
    X_seq = features[idx_test : idx_test + SEQ_LEN]
    Y_true = features[idx_test + 1 : idx_test + SEQ_LEN + 1]
    
    X_t = torch.tensor(X_seq).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        Y_pred, phys = rnn_model(X_t)
    
    Y_pred = Y_pred.squeeze(0).cpu().numpy()
    Y_true_t = Y_true  # (seq_len, 68)
    
    # Scatter dernier pas de temps (latents uniquement)
    ax = axes[1]
    ax.scatter(Y_true_t[-1, :LATENT_DIM], Y_pred[-1, :LATENT_DIM],
               alpha=0.3, s=5, color='steelblue')
    lim = [min(Y_true_t[-1, :LATENT_DIM].min(), Y_pred[-1, :LATENT_DIM].min()),
           max(Y_true_t[-1, :LATENT_DIM].max(), Y_pred[-1, :LATENT_DIM].max())]
    ax.plot(lim, lim, 'r--', label='y=x')
    ax.set_xlabel('Latent réel')
    ax.set_ylabel('Latent prédit')
    ax.set_title(f'Prédiction vs Réalité (z, dernier pas)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('RNN_01_loss_prediction.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(" RNN_01_loss_prediction.png")
    
    # --- Figure 2 : Reconstruction des signaux ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i in range(6):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        idx = 5000 + i * 100
        X_seq = features[idx : idx + SEQ_LEN]
        Y_true = features[idx + 1 : idx + SEQ_LEN + 1]
        
        X_t = torch.tensor(X_seq).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            Y_pred, _ = rnn_model(X_t)
        Y_pred = Y_pred.squeeze(0).cpu().numpy()
        
        # Dernier latent prédit vs réel
        z_pred = Y_pred[-1, :LATENT_DIM]
        z_true = Y_true[-1, :LATENT_DIM]
        stats_pred = Y_pred[-1, LATENT_DIM:]
        stats_true = Y_true[-1, LATENT_DIM:]
        
        # Décodage
        z_pt = torch.tensor(z_pred).unsqueeze(0).to(DEVICE)
        z_tt = torch.tensor(z_true).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            sig_pred_norm = ae_model.decode(z_pt).cpu().numpy()[0]
            sig_true_norm = ae_model.decode(z_tt).cpu().numpy()[0]
        
        # Dénormalisation
        sig_pred = np.zeros_like(sig_pred_norm)
        sig_true = np.zeros_like(sig_true_norm)
        sig_pred[0] = sig_pred_norm[0] * stats_pred[1] + stats_pred[0]
        sig_pred[1] = sig_pred_norm[1] * stats_pred[3] + stats_pred[2]
        sig_true[0] = sig_true_norm[0] * stats_true[1] + stats_true[0]
        sig_true[1] = sig_true_norm[1] * stats_true[3] + stats_true[2]
        
        t = np.arange(WINDOW_SIZE) * TE * 1000  # ms
        
        # Plot voie basse
        ax.plot(t, sig_true[1], color='black', lw=1.5, alpha=0.7, label='Original')
        ax.plot(t, sig_pred[1], color='coral', lw=1, linestyle='--', label='RNN prédit')
        ax.set_title(f'Exemple {i+1} (idx={idx})')
        ax.set_xlabel('ms')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    
    plt.suptitle('Reconstruction Voie Basse depuis latents prédits', fontsize=12)
    plt.tight_layout()
    plt.savefig('RNN_02_reconstruction.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(" RNN_02_reconstruction.png")
    
    # --- Figure 3 : Extraction physique (détection vallées) ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    for i in range(6):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        idx = 3000 + i * 200
        X_seq = features[idx : idx + SEQ_LEN]
        Y_true = features[idx + 1 : idx + SEQ_LEN + 1]
        
        # Utilise le vrai latent (meilleur pour la démo physique)
        z_true = Y_true[-1, :LATENT_DIM]
        stats = Y_true[-1, LATENT_DIM:]
        
        z_t = torch.tensor(z_true).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            sig_norm = ae_model.decode(z_t).cpu().numpy()[0]
        
        sig = np.zeros_like(sig_norm)
        sig[0] = sig_norm[0] * stats[1] + stats[0]
        sig[1] = sig_norm[1] * stats[3] + stats[2]
        
        t = np.arange(WINDOW_SIZE) * TE * 1000
        
        # Détection vallées
        valleys_vb = detect_valleys(sig_norm[1], 0.0, SEUIL_VALLEE)
        valleys_vh = detect_valleys(sig_norm[0], 0.0, SEUIL_VALLEE)
        
        props = estimate_drops(valleys_vb, valleys_vh, stats[2:4], stats[0:2])
        
        # Plot
        ax.plot(t, sig[1], color='steelblue', lw=1, label='Voie Basse')
        ax.plot(t, sig[0], color='coral', lw=1, alpha=0.7, label='Voie Haute')
        ax.axhline(y=stats[2], color='steelblue', linestyle='--', alpha=0.3)
        ax.axhline(y=stats[0], color='coral', linestyle='--', alpha=0.3)
        
        # Marquage vallées
        for v in valleys_vb:
            ax.axvspan(v['t_start_ms'], v['t_end_ms'], alpha=0.15, color='blue')
        for v in valleys_vh:
            ax.axvspan(v['t_start_ms'], v['t_end_ms'], alpha=0.15, color='red')
        
        title = f'Ex {i+1}: {len(valleys_vb)} vb, {len(valleys_vh)} vh'
        if props:
            title += f'\n{len(props)} gouttes:'
            for j, p in enumerate(props[:2]):
                title += f' D={p["D_mm"]:.2f}mm V={p["V_ms"]:.1f}m/s'
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('ms')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    
    plt.suptitle('Détection de vallées et estimation D, V (Loi Gunn-Kinzer)', fontsize=12)
    plt.tight_layout()
    plt.savefig('RNN_03_extraction_physique.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(" RNN_03_extraction_physique.png")
    
    # --- Figure 4 : Analyse de l'espace latent RNN ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Distribution des latents prédits vs réels
    all_z_true, all_z_pred = [], []
    for idx in range(0, len(features) - SEQ_LEN - 1, 100):
        X_seq = features[idx : idx + SEQ_LEN]
        Y_true = features[idx + 1 : idx + SEQ_LEN + 1]
        X_t = torch.tensor(X_seq).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            Y_pred, _ = rnn_model(X_t)
        all_z_true.append(Y_true[-1, :LATENT_DIM])
        all_z_pred.append(Y_pred[0, -1, :LATENT_DIM].cpu().numpy())
    
    Z_true = np.array(all_z_true)
    Z_pred = np.array(all_z_pred)
    
    # R² par dimension
    r2_dims = []
    for d in range(LATENT_DIM):
        ss_res = np.sum((Z_pred[:, d] - Z_true[:, d])**2)
        ss_tot = np.sum((Z_true[:, d] - Z_true[:, d].mean())**2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        r2_dims.append(r2)
    
    axes[0].bar(range(LATENT_DIM), r2_dims, color='steelblue', alpha=0.7)
    axes[0].axhline(y=np.mean(r2_dims), color='red', linestyle='--', 
                    label=f'Moyenne={np.mean(r2_dims):.2%}')
    axes[0].set_xlabel('Dimension du latent')
    axes[0].set_ylabel('R²')
    axes[0].set_title('R² par dimension latente')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Distribution globale des erreurs
    errors = Z_pred - Z_true
    axes[1].hist(errors.flatten(), bins=100, color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_title(f'Distribution erreurs prédiction\n(std={np.std(errors):.4f})')
    axes[1].set_xlabel('Erreur (prédit - réel)')
    axes[1].set_ylabel('Fréquence')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('RNN_04_analyse_latent.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(" RNN_04_analyse_latent.png")
    
    print("\n" + "="*50)
    print("VISUALISATIONS TERMINÉES")
    print("="*50)


# =====================================================================
# MAIN
# =====================================================================
if __name__ == '__main__':
    # 1. Charger AE
    print("="*60)
    print("ÉTAPE 2 : RNN SUR ESPACE LATENT")
    print("="*60)
    
    ae = AutoEncoder().to(DEVICE)
    if not os.path.exists(AE_WEIGHTS):
        print(f"ERREUR: {AE_WEIGHTS} introuvable !")
        pth_files = [f for f in os.listdir('.') if f.endswith('.pth')]
        if pth_files:
            print(f"Fichiers trouvés : {pth_files}")
        sys.exit(1)
    
    ae.load_state_dict(torch.load(AE_WEIGHTS, map_location=DEVICE))
    print(f"✓ AE chargé : {AE_WEIGHTS}")
    
    # 2. Extraire features
    features = extract_latent_features(ae, FICHIER_BIN, DEVICE)
    
    # 3. Entraîner RNN
    rnn, train_losses, val_losses, test_mse = train_rnn(features)
    
    # 4. Visualiser
    visualize_all(rnn, ae, features, train_losses, val_losses)
    
    print("\n" + "="*60)
    print("FIN DE L'ÉTAPE 2")
    print("="*60)