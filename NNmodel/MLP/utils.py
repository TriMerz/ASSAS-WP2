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
        self.hidden_dim = 2048  # Doubled from 1024
        
        # Enhanced encoder with more capacity
        self.encoder = nn.Sequential(
            nn.LayerNorm(self.n_features),
            
            # Wider initial transformation
            nn.Linear(self.n_features, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            
            # Multiple dense layers with residual connections
            ResidualBlock(self.hidden_dim * 2, self.hidden_dim * 2),
            ResidualBlock(self.hidden_dim * 2, self.hidden_dim * 2),
            ResidualBlock(self.hidden_dim * 2, self.hidden_dim * 2),
            
            # Deeper LSTM with larger capacity
            nn.LSTM(input_size=self.hidden_dim * 2,
                   hidden_size=self.hidden_dim,  # Doubled hidden size
                   num_layers=6,  # Doubled layers
                   bidirectional=True,
                   batch_first=True,
                   dropout=0.1),
            
            # Multiple projection layers for embedding
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim)
        )

        self.decoder = nn.Sequential(
            # Multiple expansion layers
            nn.Linear(self.embedding_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            
            # Deeper LSTM for reconstruction
            nn.LSTM(input_size=self.hidden_dim * 2,
                   hidden_size=self.hidden_dim * 2,
                   num_layers=6,  # Doubled layers
                   batch_first=True,
                   dropout=0.1),
                   
            # Multiple residual blocks
            ResidualBlock(self.hidden_dim * 2, self.hidden_dim * 2),
            ResidualBlock(self.hidden_dim * 2, self.hidden_dim * 2),
            ResidualBlock(self.hidden_dim * 2, self.hidden_dim * 2),
            
            # Final reconstruction layers
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.n_features),
            nn.LayerNorm(self.n_features)
        )

    def encode(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        # Forward through encoder layers
        for i in range(6):  # First 6 layers before LSTM
            x = self.encoder[i](x)
        # LSTM layer
        x, (hidden, _) = self.encoder[6](x)
        # Concatenate final hidden states
        hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        # Forward through remaining layers
        for i in range(7, len(self.encoder)):
            hidden_cat = self.encoder[i](hidden_cat)
            
        return hidden_cat

    def decode(self, embedding):
        x = embedding
        # Forward through initial layers
        for i in range(10):  # Layers before LSTM
            x = self.decoder[i](x)
        # Expand for sequence length
        x = x.unsqueeze(1).repeat(1, self.window_size, 1) 
        # LSTM layer
        x, _ = self.decoder[10](x)
        # Forward through remaining layers
        for i in range(11, len(self.decoder)):
            x = self.decoder[i](x)
        return x

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


class Performance(nn.Module):
    def __init__(self, input_size, embedding_dim, device):
        super(Performance, self).__init__()
        self.window_size, self.n_features = input_size
        self.embedding_dim = embedding_dim
        self.device = device
        self.hidden_dim = 2048  # Doubled from 1024

        # Enhanced encoder pathway
        self.encoder = nn.Sequential(
            # Initial processing
            nn.LayerNorm(self.n_features),
            nn.Linear(self.n_features, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            
            # Multiple transformer blocks
            TransformerBlock(d_model=self.hidden_dim * 2, nhead=16, dropout=0.1),
            TransformerBlock(d_model=self.hidden_dim * 2, nhead=16, dropout=0.1),
            TransformerBlock(d_model=self.hidden_dim * 2, nhead=16, dropout=0.1),
            TransformerBlock(d_model=self.hidden_dim * 2, nhead=16, dropout=0.1),
            
            # Multiple residual blocks
            ResidualBlock(self.hidden_dim * 2, self.hidden_dim * 2),
            ResidualBlock(self.hidden_dim * 2, self.hidden_dim * 2),
            ResidualBlock(self.hidden_dim * 2, self.hidden_dim * 2),
            
            # Reshape for enhanced 1D convolution
            Rearrange('b t f -> b f t'),
            
            # Deeper convolutional processing
            nn.Conv1d(self.hidden_dim * 2, 1024, kernel_size=3, padding=1),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Conv1d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            
            # Global context
            nn.AdaptiveAvgPool1d(1),
            Rearrange('b c 1 -> b c'),
            
            # Multiple projection layers for embedding
            nn.Linear(256, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim)
        ).to(self.device)

        # Enhanced decoder pathway
        self.decoder = nn.Sequential(
            # Multiple expansion layers
            nn.Linear(self.embedding_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            
            # Expand for sequence length
            Rearrange('b f -> b 1 f'),
            RepeatTime(self.window_size),
            
            # Multiple transformer blocks
            TransformerBlock(d_model=self.hidden_dim * 2, nhead=16, dropout=0.1),
            TransformerBlock(d_model=self.hidden_dim * 2, nhead=16, dropout=0.1),
            TransformerBlock(d_model=self.hidden_dim * 2, nhead=16, dropout=0.1),
            TransformerBlock(d_model=self.hidden_dim * 2, nhead=16, dropout=0.1),
            
            # Multiple residual blocks
            ResidualBlock(self.hidden_dim * 2, self.hidden_dim * 2),
            ResidualBlock(self.hidden_dim * 2, self.hidden_dim * 2),
            ResidualBlock(self.hidden_dim * 2, self.hidden_dim * 2),
            
            # Final reconstruction
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.n_features),
            nn.LayerNorm(self.n_features)
        ).to(self.device)

    # encode, decode, and forward methods remain the same
    def encode(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        return self.encoder(x)

    def decode(self, embedding):
        return self.decoder(embedding)

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        embedding = self.encode(x)
        reconstruction = self.decode(embedding)
        
        return embedding, reconstruction


class Rearrange(nn.Module):
    def __init__(self, pattern):
        super().__init__()
        self.pattern = pattern

    def forward(self, x):
        if self.pattern == 'b t f -> b f t':
            return x.permute(0, 2, 1)
        elif self.pattern == 'b f t -> b t f':
            return x.permute(0, 2, 1)
        elif self.pattern == 'b f -> b 1 f':
            return x.unsqueeze(1)
        elif self.pattern == 'b c 1 -> b c':
            return x.squeeze(-1)  # Rimuove l'ultima dimensione se è 1
        else:
            raise ValueError(f"Pattern {self.pattern} not supported")

class RepeatTime(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size

    def forward(self, x):
        # x shape: (batch, 1, features)
        return x.repeat(1, self.window_size, 1)
    

"""
class Performance(nn.Module):
    def __init__(self, input_size, embedding_dim, device):  # Keep same signature as Default
        super(Performance, self).__init__()
        # Unpack input dimensions like Default encoder
        self.window_size, self.n_features = input_size  # Unpack tuple
        self.embedding_dim = embedding_dim
        self.device = device
        self.hidden_dim = 1024  # Same as Default encoder

        # Encoder pathway maintaining same input format as Default
        self.encoder = nn.Sequential(
            # Initial processing
            nn.LayerNorm(self.n_features),
            nn.Linear(self.n_features, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            # Add transformer blocks for better feature interaction
            TransformerBlock(d_model=self.hidden_dim, nhead=8, dropout=0.1),
            TransformerBlock(d_model=self.hidden_dim, nhead=8, dropout=0.1),
            
            # Add residual connections for better gradient flow
            ResidualBlock(self.hidden_dim, self.hidden_dim),
            
            # Convolutional processing for local patterns
            nn.Unflatten(2, (1, self.hidden_dim)),
            ConvBlock(1, 64, kernel_size=3, padding=1),
            ConvBlock(64, 32, kernel_size=3, padding=1),
            nn.Flatten(2),
            
            # Final embedding projection
            nn.Linear(self.hidden_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim)
        ).to(self.device)

        # Decoder pathway
        self.decoder = nn.Sequential(
            # Initial processing
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            # Transformer blocks for reconstruction
            TransformerBlock(d_model=self.hidden_dim, nhead=8, dropout=0.1),
            TransformerBlock(d_model=self.hidden_dim, nhead=8, dropout=0.1),
            
            # Residual connections
            ResidualBlock(self.hidden_dim, self.hidden_dim),
            
            # Convolutional processing
            nn.Unflatten(2, (1, self.hidden_dim)),
            ConvBlock(1, 64, kernel_size=3, padding=1),
            ConvBlock(64, 32, kernel_size=3, padding=1),
            nn.Flatten(2),
            
            # Final reconstruction
            nn.Linear(self.hidden_dim, self.n_features),
            nn.LayerNorm(self.n_features)
        ).to(self.device)

    def encode(self, x):
        # Handle input like Default encoder
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        # Forward through encoder
        x = self.encoder[0](x)  # LayerNorm
        x = self.encoder[1](x)  # Linear
        x = self.encoder[2](x)  # LayerNorm
        x = self.encoder[3](x)  # GELU
        x = self.encoder[4](x)  # Dropout
        
        # Transformer and residual processing
        x = self.encoder[5](x)  # First transformer
        x = self.encoder[6](x)  # Second transformer
        x = self.encoder[7](x)  # Residual
        
        # Convolutional processing
        x = self.encoder[8:12](x)  # Conv blocks
        
        # Final embedding
        x = self.encoder[12](x)  # Linear
        embedding = self.encoder[13](x)  # LayerNorm
        
        return embedding

    def decode(self, embedding):
        # Similar structure to Default decoder
        x = self.decoder[0](embedding)  # Linear
        x = self.decoder[1](x)  # LayerNorm
        x = self.decoder[2](x)  # GELU
        x = self.decoder[3](x)  # Dropout
        
        # Expand for sequence length like Default
        x = x.unsqueeze(1).repeat(1, self.window_size, 1)
        
        # Transformer and residual processing
        x = self.decoder[4](x)  # First transformer
        x = self.decoder[5](x)  # Second transformer
        x = self.decoder[6](x)  # Residual
        
        # Convolutional processing
        x = self.decoder[7:11](x)  # Conv blocks
        
        # Final reconstruction
        x = self.decoder[11](x)  # Linear
        reconstruction = self.decoder[12](x)  # LayerNorm
        
        return reconstruction

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        embedding = self.encode(x)
        reconstruction = self.decode(embedding)
        
        return embedding, reconstruction
"""

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
        
        # Initialize encoder based on selected architecture
        input_dims = (self.window_size, self.n_features)  # Create input dimensions tuple
        
        if default:
            self._default_encoder(input_dims)
        else:
            self._performance_encoder(input_dims)
            
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
        
        criterion = CombinedLoss(alpha=0.5,    # Bilanciamento tra ricostruzione e temporale
                                 beta=0.15,    # Peso per correlazioni tra features
                                 gamma=0.25     # Enfasi su eventi rari
                                 ).to(self.device)
        
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

    def _default_encoder(self, input_dims):
        """
        Initialize default encoder architecture
        """

        self.encoder = Default(
            input_size=input_dims,
            embedding_dim=self.embedding_dim
        ).to(self.device)
        
        self.decoder = Default(
            input_size=input_dims,
            embedding_dim=self.embedding_dim
        ).to(self.device)
            
        # Optimize memory usage
        torch.cuda.empty_cache()

    def _performance_encoder(self, input_dims):
        """
        Initialize performance encoder architecture
        """

        self.encoder = Performance(
            input_size=input_dims,
            embedding_dim=self.embedding_dim,
            device=self.device
        ).to(self.device)
        
        self.decoder = Performance(
            input_size=input_dims,
            embedding_dim=self.embedding_dim,
            device=self.device
        ).to(self.device)
        
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
                    embedding, _ = self.encoder.encode(batch)
                    
            embeddings.append(embedding.cpu().numpy())
            
        return np.concatenate(embeddings, axis=0)