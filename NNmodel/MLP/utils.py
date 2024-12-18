#!/usr/bin/env python3

# Standard library imports
import os
import sys

# Third party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from scipy.fft import fft
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm
import matplotlib.pyplot as plt
from loss import *
from pathlib import Path



class TimeSeriesAugmenter(BaseEstimator, TransformerMixin):
    """Time series data augmentation transformer.
    
    Applies various transformations to increase time series variability including
    time shifts, scaling, noise addition, masking and mixing.

    Parameters
    ----------
    time_shift_range : tuple of (float, float), default=(-3, 3)
        Range for random time shifts (min, max)
    time_scale_range : tuple of (float, float), default=(0.8, 1.2)
        Range for random scaling factors (min, max)
    noise_std : float, default=0.1
        Standard deviation of Gaussian noise
    masking_probability : float, default=0.1
        Probability of masking values
    mixing_probability : float, default=0.2
        Probability of mixing different series
    random_state : int, default=42
        Random seed for reproducibility
    """
    def __init__(
        self,
        time_shift_range: tuple = (-3, 3),
        time_scale_range: tuple = (0.8, 1.2),
        noise_std: float = 0.1,
        masking_probability: float = 0.1,
        mixing_probability: float = 0.2,
        random_state: int = 42,
    ):
        self.time_shift_range = time_shift_range
        self.time_scale_range = time_scale_range
        self.noise_std = noise_std
        self.masking_probability = masking_probability
        self.mixing_probability = mixing_probability
        self.random_state = random_state

    def fit(self, X, y=None):
        """Nessuna operazione di fitting necessaria"""
        return self

    def transform(self, X):
        """
        Applica le varie trasformazioni alle serie temporali.
        
        Args:
            X (numpy.ndarray): Input array di shape (n_samples, window_size, n_features)
        
        Returns:
            numpy.ndarray: Array trasformato di shape (n_samples, window_size, n_features)
        """
        X_transformed = []
        for x in X:
            x_transformed = self._apply_transformations(x)
            X_transformed.append(x_transformed)
        return np.array(X_transformed)

    def _apply_transformations(self, x):
        """
        Applica le singole trasformazioni a una serie temporale.
        
        Args:
            x (numpy.ndarray): Input array di shape (window_size, n_features)
        
        Returns:
            numpy.ndarray: Array trasformato di shape (window_size, n_features)
        """
        x_transformed = x.copy()

        # Shifting temporale
        time_shift = self.rng.integers(*self.time_shift_range)
        x_transformed = np.roll(x_transformed, time_shift, axis=0)
        if time_shift > 0:
            x_transformed[:time_shift] = 0
        elif time_shift < 0:
            x_transformed[time_shift:] = 0

        # Scaling temporale
        time_scale = self.rng.uniform(*self.time_scale_range)
        x_transformed = np.interp(np.linspace(0, x.shape[0]-1, int(x.shape[0]*time_scale)), 
                                 np.arange(x.shape[0]), x_transformed)

        # Rumore additivo
        x_transformed += self.rng.normal(0, self.noise_std, x_transformed.shape)

        # Mascheramento features
        mask = self.rng.binomial(1, self.masking_probability, x_transformed.shape[1]) > 0
        x_transformed[:, mask] = 0

        # Mixing di finestre
        if self.rng.random() < self.mixing_probability:
            other_idx = self.rng.choice(len(x), 1)[0]
            other_x = x[other_idx]
            mix_ratio = self.rng.uniform(0.2, 0.8)
            x_transformed = (x_transformed * mix_ratio + other_x * (1 - mix_ratio))

        return x_transformed


class TimeEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.time2vec = nn.Sequential(
            nn.Linear(3, embedding_dim),  # 3 inputs: absolute time, relative position, and delta time
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def forward(self, times):
        # times shape: (batch, window_size)
        
        # Calculate relative positions within the window [0,1]
        window_start = times[:, 0].unsqueeze(1)
        window_duration = (times[:, -1] - times[:, 0]).unsqueeze(1)
        relative_pos = (times - window_start) / (window_duration + 1e-8)
        # Calculate time deltas between consecutive points
        time_deltas = torch.diff(times, dim=1)  # shape: (batch, window_size-1)
        # Pad the last position to maintain shape
        time_deltas = torch.cat([time_deltas, time_deltas[:, -1:]], dim=1)
        # Stack all time features
        time_features = torch.stack([times, relative_pos, time_deltas], dim=-1)
        
        return self.time2vec(time_features)  # (batch, window_size, embedding_dim)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.norm1 = nn.LayerNorm(out_features)
        self.norm2 = nn.LayerNorm(out_features)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        
        # Initialize projection layer only if dimensions differ
        if in_features != out_features:
            self.project = nn.Linear(in_features, out_features)
        else:
            self.project = None
        
    def forward(self, x):
        # Print shapes for debugging
        # print(f"ResidualBlock input shape: {x.shape}")
        
        # Project input if necessary
        identity = self.project(x) if self.project is not None else x
        
        # First linear layer + normalization + activation
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Second linear layer + normalization
        out = self.linear2(out)
        out = self.norm2(out)
        
        # Add residual connection
        out = out + identity
        
        # Final activation
        out = self.activation(out)
        
        # print(f"ResidualBlock output shape: {out.shape}")
        return out


class Default(nn.Module):
    def __init__(self, input_size, embedding_dim):
        super().__init__()
        self.window_size, self.n_features = input_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = 1024

        self.encoder = nn.Sequential(
            nn.LayerNorm(self.n_features),
            nn.Linear(self.n_features, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LSTM(input_size=self.hidden_dim,
                   hidden_size=self.hidden_dim // 2,  # Diviso 2 perché bidirezionale
                   num_layers=3,
                   bidirectional=True,
                   batch_first=True,
                   dropout=0.1),
            # proiezione lineare per l'embedding
            nn.Linear(self.hidden_dim, self.embedding_dim),  # da hidden_dim a embedding_dim
            nn.LayerNorm(self.embedding_dim),  # Normalizzazione della dimensione corretta
            nn.Linear(self.embedding_dim, self.embedding_dim),  # mantiene la dimensione embedding
            nn.LayerNorm(self.embedding_dim)  # Normalizzazione finale
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LSTM(input_size=self.hidden_dim,
                   hidden_size=self.hidden_dim,
                   num_layers=3,
                   batch_first=True,
                   dropout=0.1),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.n_features),
            nn.LayerNorm(self.n_features)
        )

    def encode(self, x):
        # x shape: (batch, time, features)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        # Forward through encoder layers sequentially
        x = self.encoder[0](x)  # LayerNorm
        x = self.encoder[1](x)  # Linear -> hidden_dim
        x = self.encoder[2](x)  # LayerNorm
        x = self.encoder[3](x)  # GELU
        x = self.encoder[4](x)  # Dropout
        x, (hidden, _) = self.encoder[5](x)  # LSTM
        # Concatenate final hidden states
        hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)  # -> hidden_dim
        x = self.encoder[6](hidden_cat)  # Linear -> embedding_dim
        x = self.encoder[7](x)  # LayerNorm -> embedding_dim
        x = self.encoder[8](x)  # Linear -> embedding_dim
        embedding = self.encoder[9](x)  # LayerNorm -> embedding_dim
        
        return embedding

    def decode(self, embedding):
        # embedding shape: (batch, embedding_dim)
        x = self.decoder[0](embedding)  # Linear -> hidden_dim
        x = self.decoder[1](x)  # LayerNorm
        x = self.decoder[2](x)  # GELU
        x = self.decoder[3](x)  # Dropout
        
        # Expand for sequence length
        x = x.unsqueeze(1).repeat(1, self.window_size, 1)
        
        x, _ = self.decoder[4](x)  # LSTM
        x = self.decoder[5](x)  # LayerNorm
        x = self.decoder[6](x)  # Linear -> n_features
        reconstruction = self.decoder[7](x)  # LayerNorm
        
        return reconstruction

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        embedding = self.encode(x)
        reconstruction = self.decode(embedding)
        
        return embedding, reconstruction

"""
class Defaul_BIG(nn.Module):
    def __init__(self, input_size, embedding_dim):
        super().__init__()
        self.window_size, self.n_features = input_size
        self.embedding_dim = embedding_dim
        
        # Aumentiamo significativamente le dimensioni nascoste
        self.hidden_dim = 2048  # Molto più grande
        self.num_layers = 4     # Più layers
        
        self.encoder = nn.Sequential(
            # 1. Normalizzazione iniziale e proiezione
            nn.LayerNorm(self.n_features),
            nn.Linear(self.n_features, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            
            # 2. Stack di LSTM bidirezionali
            BidirectionalLSTMStack(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=0.2
            ),
            
            # 3. Post-LSTM processing
            nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),  # *2 per bidirezionale
                nn.GELU(),
                nn.Dropout(0.2),
                ResidualBlock(self.hidden_dim, self.hidden_dim),
                ResidualBlock(self.hidden_dim, self.hidden_dim)
            ),
            
            # 4. Final embedding projection with skip connection
            EmbeddingProjection(self.hidden_dim, self.embedding_dim)
        )

        self.decoder = nn.Sequential(
            # 1. Initial projection
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            
            # 2. Stack di LSTM decoder
            DecoderLSTMStack(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=0.2
            ),
            
            # 3. Post-processing con residual connections
            nn.Sequential(
                ResidualBlock(self.hidden_dim, self.hidden_dim),
                ResidualBlock(self.hidden_dim, self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.2)
            ),
            
            # 4. Final reconstruction
            ReconstructionLayer(self.hidden_dim // 2, self.n_features)
        )

    def encode(self, x):
        # x shape: (batch, time, features)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        
        # Validazione dimensioni
        assert len(x.shape) == 3, f"Input deve essere 3D, got shape {x.shape}"
        assert x.shape[1] == self.window_size, f"Expected window size {self.window_size}, got {x.shape[1]}"
        assert x.shape[2] == self.n_features, f"Expected features {self.n_features}, got {x.shape[2]}"
        
        return self.encoder(x)

    def decode(self, embedding):
        # embedding shape: (batch, embedding_dim)
        assert len(embedding.shape) == 2, f"Embedding deve essere 2D, got shape {embedding.shape}"
        assert embedding.shape[1] == self.embedding_dim, f"Expected embedding dim {self.embedding_dim}, got {embedding.shape[1]}"
        
        # Repeat embedding for each timestep
        x = embedding.unsqueeze(1).repeat(1, self.window_size, 1)
        return self.decoder(x)

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        embedding = self.encode(x)
        reconstruction = self.decode(embedding)
        
        return embedding, reconstruction

class BidirectionalLSTMStack(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm_stack = nn.ModuleList([
            nn.LSTM(
                input_size=input_size if i == 0 else hidden_size * 2,
                hidden_size=hidden_size,
                bidirectional=True,
                batch_first=True,
                dropout=dropout if i < num_layers - 1 else 0
            )
            for i in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_size * 2)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for lstm, norm in zip(self.lstm_stack, self.norms):
            # Residual connection se le dimensioni corrispondono
            residual = x if x.size(-1) == lstm.hidden_size * 2 else None
            x, _ = lstm(x)
            x = norm(x)
            if residual is not None:
                x = x + residual
        return x

class DecoderLSTMStack(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm_stack = nn.ModuleList([
            nn.LSTM(
                input_size=input_size if i == 0 else hidden_size,
                hidden_size=hidden_size,
                batch_first=True,
                dropout=dropout if i < num_layers - 1 else 0
            )
            for i in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_size)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for lstm, norm in zip(self.lstm_stack, self.norms):
            residual = x if x.size(-1) == lstm.hidden_size else None
            x, _ = lstm(x)
            x = norm(x)
            if residual is not None:
                x = x + residual
        return x

class EmbeddingProjection(nn.Module):
    def __init__(self, hidden_dim, embedding_dim):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
    def forward(self, x):
        # Global average pooling sulla dimensione temporale
        x = x.mean(dim=1)
        return self.project(x)

class ReconstructionLayer(nn.Module):
    def __init__(self, hidden_dim, n_features):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, n_features),
            nn.LayerNorm(n_features)
        )
        
    def forward(self, x):
        return self.project(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(out_features, out_features),
            nn.LayerNorm(out_features)
        )
        
        self.downsample = None if in_features == out_features else nn.Linear(in_features, out_features)
        
    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        return self.net(x) + identity
"""

class Performance(nn.Module):
    def __init__(self, n_features, window_size, embedding_dim, device):
        super(Performance, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.device = device

        self.time_encoder = TimeEncoder(embedding_dim=64).to(self.device)
        
        # Encoder pathway
        self.encoder = nn.Sequential(
            ConvBlock((self.n_features - 1) + 64, 128),  # -1 for time feature, +64 for time encoding
            nn.Flatten(),
            ResidualBlock(128 * self.window_size, 512),
            TransformerBlock(d_model=512, nhead=8),
            nn.Linear(512, self.embedding_dim)
        ).to(self.device)

        # Decoder pathway - reconstruct full input including time features
        self.decoder = nn.Sequential(
            nn.Linear(self.embedding_dim, 512),
            TransformerBlock(d_model=512, nhead=8),
            ResidualBlock(512, 128 * self.window_size),
            nn.Unflatten(1, (128, self.window_size)),
            ConvBlock(128, self.n_features)  # Reconstruct all features including time
        ).to(self.device)

    def encode(self, x):
        # x shape: (batch, features, timesteps)
        batch_size, n_features, time_steps = x.shape
        
        times = x[:, 0, :]  # (batch, timesteps)
        features = x[:, 1:, :]  # (batch, n_features-1, timesteps)
        
        # Encode time
        time_encoding = self.time_encoder(times)  # (batch, timesteps, 64)
        time_encoding = time_encoding.transpose(1, 2)  # (batch, 64, timesteps)
        
        # Concatenate time encoding with other features
        x_combined = torch.cat([time_encoding, features], dim=1)
        
        # L'encoder dovrebbe ora produrre (batch, embedding_dim)
        embedding = self.encoder(x_combined)
        return embedding

    def decode(self, embedding):
        """
        Decode the embedding back to original space
        """
        # Reshape for decoder
        x = self.decoder(embedding)
        return x

    def forward(self, x):
        """
        Full forward pass:
            - x.shape = (batch, n_features, window_size)
        """
        # Check input type and shape
        if isinstance(x, tuple):
            x = x[0]
        if not isinstance(x, torch.Tensor):     # Ensure x is a tensor
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
            
        # If input is 2D, add batch dimension
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        # Reshape if necessary: (batch, window_size * n_features) -> (batch, n_features, window_size)
        if x.shape[1] == self.window_size * self.n_features:
            x = x.view(x.shape[0], self.n_features, self.window_size)
            
        # Get embedding
        embedding = self.encode(x)
        # Get reconstruction
        reconstruction = self.decode(embedding)
        
        return embedding, reconstruction


class NonLinearEmbedder:
    """Optimized version of NonLinearEmbedder with better performance and memory management"""
    def __init__(self,
                n_features: int,
                checkpoint_dir: str,
                window_size: int,
                default: bool,
                embedding_dim: int = 256,
                device: str = None):
        """
        Initialize embedder with optimized settings
        """
        self.n_features = n_features
        self.window_size = window_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.default = default
        self.embedding_dim = embedding_dim
        
        # Optimize device selection
        if device is None:
            if torch.cuda.is_available():
                # Get the GPU with most free memory
                gpu_id = torch.cuda.current_device()
                torch.cuda.set_device(gpu_id)
                self.device = f'cuda:{gpu_id}'
            else:
                self.device = 'cpu'
        else:
            self.device = device
            
        # Initialize encoder/decoder
        self.encoder = None
        self.decoder = None
        self.encoder_path = self.checkpoint_dir / "encoder.pth"
        self.decoder_path = self.checkpoint_dir / "decoder.pth"
        
        # Initialize based on mode
        if default:
            self._default_encoder(n_features * window_size)
        else:
            self._performance_encoder()
            
        # Move models to device
        self.to_device()
        
        # Initialize training components
        self.history = {'train_loss': [], 'val_loss': []}
        self.scaler = GradScaler('cuda')  # For mixed precision training
        self.augmenter = TimeSeriesAugmenter()
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def to_device(self):
        """Move models to device efficiently"""
        if self.encoder:
            self.encoder = self.encoder.to(self.device)
        if self.decoder:
            self.decoder = self.decoder.to(self.device)
    
    @torch.cuda.amp.autocast('cuda')
    def fit(self,
            windows,
            epochs: int = 100,
            batch_size: int = 32,
            learning_rate: float = 1e-4,
            validation_split: float = 0.2,
            weight_decay: float = 1e-4,
            patience: int = 20,
            data_augmentation: bool = False,
            use_amp: bool = True):
        """Training function with proper history logging"""
        self.validate_input_shape(windows)
        
        # Convert input to tensor if needed
        if not isinstance(windows, torch.Tensor):
            windows = torch.FloatTensor(windows)

        # Ensure 3D shape
        if len(windows.shape) == 2:
            windows = windows.unsqueeze(0)
        
        # Data augmentation if requested
        if data_augmentation:
            windows = self.augmenter.transform(windows)
        
        # Prepare data loaders
        train_data, val_data = self._prepare_data(windows, validation_split)
        train_loader = self._create_dataloader(train_data, batch_size, shuffle=True)
        val_loader = self._create_dataloader(val_data, batch_size, shuffle=False)
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler initialization
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=len(train_loader)
        )
        
        criterion = CombinedLoss(alpha=0.7).to(self.device)
        best_val_loss = float('inf')
        patience_counter = 0
        self.history = {'train_loss': [], 'val_loss': []}  # Reset history at start of training
        
        print(f"\nStarting training for {epochs} epochs:")
        print(f"{'Epoch':>5} {'Train Loss':>12} {'Val Loss':>12} {'Best':>6}")
        print("-" * 40)
        
        # Training loop
        for epoch in range(epochs):
            train_loss = self._train_epoch(
                train_loader,
                optimizer,
                criterion,
                scheduler,
                use_amp
            )
            
            val_loss = self._validate_epoch(val_loader, criterion)
            
            # Log losses
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            is_best = val_loss < best_val_loss
            best_marker = "*" if is_best else ""
            
            print(f"{epoch+1:5d} {train_loss:12.6f} {val_loss:12.6f} {best_marker:>6}")
            
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint()
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                self._load_checkpoint()
                break
        
        # Plot training history
        self.plot_training_history()

    def _train_epoch(self, train_loader, optimizer, criterion, scheduler, use_amp):
        """Enhanced training with debugging info"""
        self.encoder.train()
        self.decoder.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            batch = batch.to(self.device)
            
            optimizer.zero_grad()
            
            try:
                if use_amp and torch.cuda.is_available():
                    with torch.amp.autocast(device_type='cuda'):
                        # Forward pass
                        embedding = self.encoder.encode(batch)
                        reconstruction = self.decoder.decode(embedding)
                        loss = criterion(reconstruction, batch)
                    
                    # Print debug info for first batch
                    if batch_idx == 0:
                        print(f"\nDebug info for first batch:")
                        print(f"Input range: [{batch.min():.3f}, {batch.max():.3f}]")
                        print(f"Embedding range: [{embedding.min():.3f}, {embedding.max():.3f}]")
                        print(f"Reconstruction range: [{reconstruction.min():.3f}, {reconstruction.max():.3f}]")
                        print(f"Loss: {loss.item():.6f}")
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    # Forward pass
                    embedding = self.encoder.encode(batch)
                    reconstruction = self.decoder.decode(embedding)
                    loss = criterion(reconstruction, batch)
                    loss.backward()
                    optimizer.step()
                
                scheduler.step()
                total_loss += loss.item()
                        
            except RuntimeError as e:
                print(f"\nError in batch {batch_idx + 1}:")
                print(f"Input shape: {batch.shape}")
                print(f"Device: {self.device}")
                print(f"Error: {str(e)}")
                raise e

        return total_loss / num_batches

    @torch.no_grad()
    def _validate_epoch(self, val_loader, criterion):
        """Optimized validation epoch"""
        self.encoder.eval()
        self.decoder.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (tuple, list)):
                    batch = batch[0]
                batch = batch.to(self.device)
                
                # Forward pass
                embedding = self.encoder.encode(batch)
                reconstruction = self.decoder.decode(embedding)
                loss = criterion(reconstruction, batch)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def _prepare_data(self, windows, validation_split):
        """Prepare data for training efficiently"""
        n_val = int(len(windows) * validation_split)
        indices = torch.randperm(len(windows))
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]
        
        if isinstance(windows, torch.Tensor):
            train_data = windows[train_indices]
            val_data = windows[val_indices]
        else:
            train_data = torch.FloatTensor(windows[train_indices])
            val_data = torch.FloatTensor(windows[val_indices])
        
        return train_data, val_data
    
    def _create_dataloader(self, data, batch_size, shuffle):
        """Create optimized DataLoader"""
        return DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
    
    def _save_checkpoint(self):
        """Save checkpoint with both encoder and decoder states"""
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            
        # Save encoder
        encoder_path = os.path.join(self.checkpoint_dir, "encoder.pth")
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'history': self.history
        }, encoder_path)
        
        # Save decoder
        decoder_path = os.path.join(self.checkpoint_dir, "decoder.pth")
        torch.save({
            'decoder_state_dict': self.decoder.state_dict(),
            'history': self.history
        }, decoder_path)
        
        print(f"Saved encoder to {encoder_path}")
        print(f"Saved decoder to {decoder_path}")

    def _load_checkpoint(self):
        """Load both encoder and decoder checkpoints"""
        encoder_path = os.path.join(self.checkpoint_dir, "encoder.pth")
        decoder_path = os.path.join(self.checkpoint_dir, "decoder.pth")
        
        if os.path.exists(encoder_path) and os.path.exists(decoder_path):
            # Load encoder
            encoder_checkpoint = torch.load(encoder_path, map_location=self.device)
            self.encoder.load_state_dict(encoder_checkpoint['encoder_state_dict'])
            self.history = encoder_checkpoint['history']
            
            # Load decoder
            decoder_checkpoint = torch.load(decoder_path, map_location=self.device)
            self.decoder.load_state_dict(decoder_checkpoint['decoder_state_dict'])
            
            print("Loaded checkpoints successfully")
        else:
            raise FileNotFoundError(f"Checkpoint files not found in {self.checkpoint_dir}")

    def _default_encoder(self, input_size):
        """
        Build optimized encoder architecture for high-dimensional non-linear data.
        Args:
            input_size: nel caso 3D sarà window_size * n_features
        """
        if self.default:
            # Calcoliamo n_features dalla dimensione di input
            n_features = input_size // self.window_size

            # Creiamo una tupla (window_size, n_features) per l'inizializzazione
            input_dims = (self.window_size, n_features)

            # Inizializziamo l'encoder con le dimensioni corrette
            self.encoder = Default(
                input_size=input_dims,
                embedding_dim=self.embedding_dim
            ).to(self.device)
            
            # Il decoder avrà la stessa struttura
            self.decoder = Default(
                input_size=input_dims,
                embedding_dim=self.embedding_dim
            ).to(self.device)
            
            # Optimize memory usage
            torch.cuda.empty_cache()

    def _performance_encoder(self):
        """
        Build optimized encoder architecture for time series with irregular intervals.
        """
        self.encoder = Performance(n_features=self.n_features,
                                window_size=self.window_size,
                                embedding_dim=self.embedding_dim,
                                device=self.device)
        self.decoder = Performance(n_features=self.n_features,
                                window_size=self.window_size,
                                embedding_dim=self.embedding_dim,
                                device=self.device)
        
        # Optimize memory usage
        torch.cuda.empty_cache()

    def visualize_embeddings(self, windows, labels=None):
        """
        Embedding visualization using t-SNE
        """
        # Get embeddings efficiently using batch processing
        embeddings = self.transform(windows)
        
        # Reduce dimensionality for visualization
        n_samples = embeddings.shape[0]
        perplexity = min(30, n_samples - 1)
        
        # CPU version of t-SNE
        tsne = TSNE(n_components=2,
                    random_state=42,
                    perplexity=perplexity,
                    n_jobs=-1)  # Use all available CPU cores
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        if labels is not None:
            scatter = plt.scatter(embeddings_2d[:, 0],
                                embeddings_2d[:, 1],
                                c=labels,
                                cmap='viridis',
                                alpha=0.6)
            plt.colorbar(scatter)
        else:
            plt.scatter(embeddings_2d[:, 0],
                    embeddings_2d[:, 1],
                    alpha=0.6)
        
        plt.title('t-SNE visualization of embeddings')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        
        # Save plot
        plot_path = self.checkpoint_dir / 'tsne_embeddings.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"t-SNE Plot saved at: {plot_path}")

    def plot_training_history(self):
        """Enhanced training history visualization"""
        if not hasattr(self, 'history') or not self.history:
            print("No training history available to plot")
            return
            
        plt.figure(figsize=(12, 6))
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Plot losses
        plt.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        
        # Add moving average for smoothing
        window = min(5, len(epochs) // 10)  # Dynamic window size
        if window > 1:
            train_ma = np.convolve(self.history['train_loss'], 
                                np.ones(window)/window, 
                                mode='valid')
            val_ma = np.convolve(self.history['val_loss'], 
                                np.ones(window)/window, 
                                mode='valid')
            ma_epochs = epochs[window-1:]
            plt.plot(ma_epochs, train_ma, 'b--', alpha=0.5, label='Train MA')
            plt.plot(ma_epochs, val_ma, 'r--', alpha=0.5, label='Val MA')
        
        # Enhance plot
        plt.grid(True, alpha=0.3)
        plt.title('Training and Validation Loss Over Time', pad=20)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Use log scale if loss values vary significantly
        if max(self.history['train_loss']) / min(self.history['train_loss']) > 100:
            plt.yscale('log')
        
        # Add min/max annotations
        min_train = min(self.history['train_loss'])
        min_val = min(self.history['val_loss'])
        plt.annotate(f'Min Train: {min_train:.6f}', 
                    xy=(epochs[self.history['train_loss'].index(min_train)], min_train),
                    xytext=(10, 10), textcoords='offset points')
        plt.annotate(f'Min Val: {min_val:.6f}',
                    xy=(epochs[self.history['val_loss'].index(min_val)], min_val),
                    xytext=(10, -10), textcoords='offset points')
        
        # Save plot
        plot_path = self.checkpoint_dir / 'training_history.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nTraining history plot saved to {plot_path}")

    def checkpoint_exists(self):
        """Check if checkpoint exists with proper error handling"""
        try:
            return self.encoder_path.exists() and self.decoder_path.exists()
        except Exception as e:
            print(f"Error checking checkpoint existence: {e}")
            return False

    def evaluate_reconstruction(self, windows, n_samples=5):
        """
        Reconstruction evaluation with equal weights for all points
        """
        print(f"\nInput windows shape: {windows.shape}")
        emb = self.transform(windows)
        print(f"Embeddings shape: {emb.shape}")
        reconstructed = self.inverse_transform(emb)
        print(f"Reconstructed shape: {reconstructed.shape}")
        
        if self.default:
            # Reshape windows for processing
            try:
                windows_reshaped = windows.reshape(windows.shape[0], self.window_size, -1)
                reconstructed_reshaped = reconstructed.reshape(reconstructed.shape[0], self.window_size, -1)
            except ValueError:
                windows_reshaped = windows.copy().reshape(windows.shape[0], self.window_size, -1)
                reconstructed_reshaped = reconstructed.copy().reshape(reconstructed.shape[0], self.window_size, -1)
            
            n_features = windows_reshaped.shape[2]
            total_length = windows.shape[0] + self.window_size - 1
            
            # Initialize arrays
            original_series = np.zeros((total_length, n_features))
            reconstructed_series = np.zeros((total_length, n_features))
            counts = np.zeros((total_length, n_features))
            
            # Process each window with equal weights
            for i in range(windows_reshaped.shape[0]):
                start_idx = i
                end_idx = start_idx + self.window_size
                
                # Add values directly without weights
                original_series[start_idx:end_idx] += windows_reshaped[i]
                reconstructed_series[start_idx:end_idx] += reconstructed_reshaped[i]
                counts[start_idx:end_idx] += 1
                
        else:
            # Handle performance encoder case
            n_features = windows.shape[1]
            total_length = windows.shape[0] + self.window_size - 1
            original_series = np.zeros((total_length, n_features))
            reconstructed_series = np.zeros((total_length, n_features))
            counts = np.zeros((total_length, n_features))
            
            for i in range(windows.shape[0]):
                start_idx = i
                end_idx = start_idx + self.window_size
                
                original_series[start_idx:end_idx] += windows[i].T
                reconstructed_series[start_idx:end_idx] += reconstructed[i].T
                counts[start_idx:end_idx] += 1
        
        # Average overlapping points
        mask = counts > 0
        original_series[mask] /= counts[mask]
        reconstructed_series[mask] /= counts[mask]
        
        # Print reconstruction stats
        print("\nReconstruction Statistics:")
        print(f"Original range: [{original_series.min():.3f}, {original_series.max():.3f}]")
        print(f"Reconstructed range: [{reconstructed_series.min():.3f}, {reconstructed_series.max():.3f}]")
        
        # Plotting
        time = np.arange(total_length)
        feature_indices = np.random.choice(n_features-1, n_samples, replace=False) + 1
        
        fig, axes = plt.subplots(n_samples, 1, figsize=(15, 4*n_samples))
        if n_samples == 1:
            axes = [axes]
        
        metrics_dict = {}
        for i, feat_idx in enumerate(feature_indices):
            # Plot
            axes[i].plot(time, original_series[:, feat_idx],
                        label='Original', color='blue', alpha=0.7, linewidth=2)
            axes[i].plot(time, reconstructed_series[:, feat_idx],
                        label='Reconstructed', color='red', alpha=0.7, linewidth=2, linestyle='--')
            
            axes[i].set_title(f'Feature {feat_idx}')
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel('Value')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            # Calculate metrics
            metrics_dict[feat_idx] = {
                'MSE': mean_squared_error(original_series[:, feat_idx], 
                                        reconstructed_series[:, feat_idx]),
                'MAE': mean_absolute_error(original_series[:, feat_idx], 
                                        reconstructed_series[:, feat_idx]),
                'Correlation': np.corrcoef(original_series[:, feat_idx],
                                        reconstructed_series[:, feat_idx])[0,1],
                'R2': r2_score(original_series[:, feat_idx],
                            reconstructed_series[:, feat_idx])
            }
        
        plt.tight_layout()
        plot_path = self.checkpoint_dir / 'reconstruction_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Print metrics
        print("\nReconstruction Metrics:")
        for feat_idx, metric_values in metrics_dict.items():
            print(f"\nFeature {feat_idx}:")
            for metric_name, value in metric_values.items():
                print(f"  {metric_name}: {value:.6f}")

    def transform(self, windows):
        """Transform data to embeddings"""
        self.encoder.eval()
        
        # Convert to tensor if needed
        if isinstance(windows, np.ndarray):
            windows = torch.FloatTensor(windows)
        
        # Add batch dimension if needed
        if len(windows.shape) == 2:
            windows = windows.unsqueeze(0)
        
        # Process in batches
        batch_size = 1024
        embeddings = []
        
        with torch.no_grad():
            try:
                for i in range(0, len(windows), batch_size):
                    batch = windows[i:i + batch_size].to(self.device)
                    embedding = self.encoder.encode(batch)
                    embeddings.append(embedding.cpu())
                    
                final_embeddings = torch.cat(embeddings, dim=0)
                return final_embeddings.numpy()
                
            except RuntimeError as e:
                print(f"\nError during transform:")
                print(f"Input shape: {windows.shape}")
                print(f"Device: {self.device}")
                print(f"Error: {str(e)}")
                raise

    def inverse_transform(self, embeddings):
        """Reconstruct from embeddings"""
        self.decoder.eval()
        
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.FloatTensor(embeddings)
        
        if len(embeddings.shape) == 1:
            embeddings = embeddings.unsqueeze(0)
        
        batch_size = 1024
        reconstructions = []
        
        with torch.no_grad():
            try:
                for i in range(0, len(embeddings), batch_size):
                    batch = embeddings[i:i + batch_size].to(self.device)
                    reconstruction = self.decoder.decode(batch)
                    reconstructions.append(reconstruction.cpu())
                    
                final_reconstructions = torch.cat(reconstructions, dim=0)
                return final_reconstructions.numpy()
                
            except RuntimeError as e:
                print(f"\nError during inverse transform:")
                print(f"Input shape: {embeddings.shape}")
                print(f"Device: {self.device}")
                print(f"Error: {str(e)}")
                raise

    def validate_input_shape(self, windows):
        """Validate input shape with improved error messages"""
        try:
            if not isinstance(windows, (np.ndarray, torch.Tensor)):
                raise ValueError("Input must be numpy array or torch tensor")
            
            # Convert to numpy for consistent handling
            if isinstance(windows, torch.Tensor):
                windows = windows.cpu().numpy()
            
            if len(windows.shape) not in [2, 3]:
                raise ValueError(f"Input must be 2D or 3D, got shape {windows.shape}")
            
            if self.default:
                if len(windows.shape) == 2:
                    # Single sample case
                    if windows.shape != (self.window_size, self.n_features):
                        raise ValueError(
                            f"Expected shape (window_size, n_features) = "
                            f"({self.window_size}, {self.n_features}), "
                            f"got {windows.shape}"
                        )
                else:  # 3D case
                    if windows.shape[1:] != (self.window_size, self.n_features):
                        raise ValueError(
                            f"Expected shape (batch, window_size, n_features) = "
                            f"(*, {self.window_size}, {self.n_features}), "
                            f"got {windows.shape}"
                        )
            else:
                # Performance encoder case
                if len(windows.shape) == 2:
                    if windows.shape[1] != self.n_features:
                        raise ValueError(
                            f"Expected shape (time_steps, n_features) = "
                            f"(*, {self.n_features}), got {windows.shape}"
                        )
                else:
                    if windows.shape[2] != self.window_size or windows.shape[1] != self.n_features:
                        raise ValueError(
                            f"Expected shape (batch, n_features, window_size) = "
                            f"(*, {self.n_features}, {self.window_size}), "
                            f"got {windows.shape}"
                        )
            
            # Check for NaN values
            if np.isnan(windows).any():
                raise ValueError("Input contains NaN values")
            
            return True
            
        except Exception as e:
            print(f"\nError during shape validation:")
            print(f"Input shape: {windows.shape if hasattr(windows, 'shape') else 'unknown'}")
            print(f"Error: {str(e)}")
            raise

    def _batch_transform(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """Transform data in batches to avoid memory issues"""
        n_samples = len(X)
        embeddings = []
        
        for i in range(0, n_samples, batch_size):
            batch = X[i:i + batch_size]
            if isinstance(batch, np.ndarray):
                batch = torch.FloatTensor(batch)
            batch = batch.to(self.device)
            
            with torch.no_grad():
                if self.default:
                    embedding = self.encoder.encode(batch)
                else:
                    embedding, _ = self.encoder(batch)
                    
            embeddings.append(embedding.cpu().numpy())
            
        return np.concatenate(embeddings, axis=0)