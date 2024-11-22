#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

# Loss function for the autoencoder
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.15, gamma=2.0):
        """
        Parameters:
        - alpha: peso per la reconstruction loss
        - beta: peso per la feature correlation loss
        - gamma: parametro focal loss per enfatizzare eventi rari
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        assert alpha + beta <= 1.0, "alpha + beta deve essere <= 1"

    def focal_mse(self, pred, target):
        """
        MSE modificata che dà più importanza agli eventi rari (valori non zero)
        """
        # Calcola MSE base
        mse = F.mse_loss(pred, target, reduction='none')
        
        # Calcola pesi per focal loss
        # Più alto il valore assoluto del target, più alto il peso
        weights = (1 + torch.abs(target)) ** self.gamma
        
        # Normalizza i pesi per stabilità
        weights = weights / (weights.mean() + 1e-6)
        
        return (weights * mse).mean()

    def temporal_consistency(self, reconstructed, original):
        """
        Penalizza inconsistenze temporali, specialmente durante le transizioni
        """
        # Calcola differenze temporali
        rec_diff = reconstructed[:, 1:] - reconstructed[:, :-1]
        orig_diff = original[:, 1:] - original[:, :-1]
        
        # Identifica transizioni significative
        is_transition = torch.abs(orig_diff) > 1e-6
        
        # Pesa più pesantemente le transizioni (es: formazione di idrogeno)
        weights = 1.0 + 4.0 * is_transition.float()
        
        # Normalizza i pesi
        weights = weights / (weights.mean() + 1e-6)
        
        return (weights * F.mse_loss(rec_diff, orig_diff, reduction='none')).mean()

    def feature_correlation_loss(self, reconstructed, original):
        """
        Mantiene le correlazioni tra features, importante per le relazioni
        tra diverse specie chimiche
        """
        # Calcola correlazioni
        rec_corr = self.batch_correlations(reconstructed)
        orig_corr = self.batch_correlations(original)
        
        # MSE sulle correlazioni
        return F.mse_loss(rec_corr, orig_corr)
    
    def physics_consistency(self, reconstructed, original):
        # Zero-state preservation (for hydrogen-like features)
        zero_mask = (original.abs() < 1e-5)
        zero_state_error = (reconstructed[zero_mask] - original[zero_mask]).abs().mean()
        
        # Preserve positive values for physical quantities
        positive_mask = (original > 0)
        positivity_error = F.relu(-reconstructed[positive_mask]).mean()
        
        # Preserve sharp transitions
        transitions = torch.abs(original[:, 1:] - original[:, :-1])
        sharp_transition_mask = (transitions > transitions.mean() + transitions.std())
        transition_error = F.mse_loss(
            reconstructed[:, 1:][sharp_transition_mask],
            original[:, 1:][sharp_transition_mask]
        )
        
        return zero_state_error + positivity_error + transition_error

    @staticmethod
    def batch_correlations(x):
        """
        Calcola la matrice di correlazione tra features
        """
        # x shape: (batch, time, features)
        b, t, f = x.shape
        
        # Reshape per il calcolo delle correlazioni
        x_flat = x.reshape(-1, f)
        
        # Normalizza
        mean = x_flat.mean(dim=0, keepdim=True)
        centered = x_flat - mean
        
        # Calcola correlazioni
        cov = centered.T @ centered / (b * t - 1)
        std = torch.sqrt(torch.diag(cov).reshape(-1, 1))
        corr = cov / (std @ std.T + 1e-8)
        
        return corr

    def forward(self, reconstructed, original):
        # Basic reconstruction loss
        rec_norm = (reconstructed - reconstructed.mean()) / (reconstructed.std() + 1e-6)
        orig_norm = (original - original.mean()) / (original.std() + 1e-6)
        mse_loss = F.mse_loss(rec_norm, orig_norm)
        
        # Enhanced temporal consistency for better spike detection
        temp_loss = self.temporal_consistency(rec_norm, orig_norm)
        
        # Feature correlation with emphasis on physical relationships
        feat_loss = self.feature_correlation_loss(rec_norm, orig_norm)
        
        # Physical constraints preservation
        physics_loss = self.physics_consistency(reconstructed, original)
        
        total_loss = (self.alpha * mse_loss + 
                     (1 - self.alpha - self.beta - self.gamma) * temp_loss + 
                     self.beta * feat_loss +
                     self.gamma * physics_loss)
        
        return total_loss