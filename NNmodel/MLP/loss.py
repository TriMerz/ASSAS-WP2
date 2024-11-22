#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

# Loss function for the autoencoder
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.15):  # Aggiungi beta nei parametri
        super().__init__()
        self.alpha = alpha
        self.beta = beta  # Definisci beta come attributo della classe
        
    def forward(self, reconstructed, original):
        # MSE loss su dati normalizzati
        rec_norm = (reconstructed - reconstructed.mean()) / (reconstructed.std() + 1e-6)
        orig_norm = (original - original.mean()) / (original.std() + 1e-6)
        mse_loss = F.mse_loss(rec_norm, orig_norm)
        
        # Temporal consistency loss
        temp_loss = self.temporal_consistency(rec_norm, orig_norm)
        
        # Feature correlation loss - normalizzato anche questo
        feat_loss = self.feature_correlation_loss(rec_norm, orig_norm)  # Usa i dati normalizzati
        
        # Verifica che i pesi sommino a 1
        assert self.alpha + self.beta <= 1.0, "alpha + beta deve essere <= 1"
        
        # Calcola la loss totale
        total_loss = (self.alpha * mse_loss + 
                     (1 - self.alpha - self.beta) * temp_loss + 
                     self.beta * feat_loss)
        
        # Debug printing (opzionale)
        if torch.isnan(total_loss):
            print(f"NaN detected! MSE: {mse_loss}, Temp: {temp_loss}, Feat: {feat_loss}")
            print(f"Norms - Rec: {reconstructed.norm()}, Orig: {original.norm()}")
        
        return total_loss
    
    def temporal_consistency(self, reconstructed, original):
        # Calculate differences between consecutive time steps
        rec_diff = reconstructed[:, 1:] - reconstructed[:, :-1]
        orig_diff = original[:, 1:] - original[:, :-1]
        
        # Normalizzazione già fatta sui dati di input
        return F.mse_loss(rec_diff, orig_diff)
    
    def feature_correlation_loss(self, reconstructed, original):
        # Calculate feature-wise correlations
        rec_corr = self.batch_correlations(reconstructed)
        orig_corr = self.batch_correlations(original)
        
        return F.mse_loss(rec_corr, orig_corr)
        
    @staticmethod
    def batch_correlations(x):
        # x shape: (batch, time, features)
        b, t, f = x.shape
        
        # Reshape to (batch * time, features)
        x_flat = x.reshape(-1, f)
        
        # Calcola correlazioni
        mean = x_flat.mean(dim=0, keepdim=True)
        centered = x_flat - mean
        cov = centered.T @ centered / (b * t - 1)
        std = torch.sqrt(torch.diag(cov).reshape(-1, 1))
        corr = cov / (std @ std.T + 1e-8)
        
        return corr