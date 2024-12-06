#!/usr/bin/env python3

# Standard library imports
import os
import sys

# Third party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm
import matplotlib.pyplot as plt
from loss import *
from pathlib import Path

from NNmodel.MLP.AEconfig import *
from Preprocessor import *
from loss import *
from AEmodels import *

config = setup_config()



class NonLinearEmbedder:
    def __init__(self,
                n_features: int,
                checkpoint_dir: str,
                window_size: int,
                embedding_dim: int = 256,
                device: str = None):
        """
        Initialize embedder with Autoencoder architecture
        """
        self.n_features = n_features
        self.window_size = window_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.embedding_dim = embedding_dim
        
        if device is None:
            if torch.cuda.is_available():
                gpu_id = torch.cuda.current_device()
                torch.cuda.set_device(gpu_id)
                self.device = f'cuda:{gpu_id}'
            else:
                self.device = 'cpu'
        else:
            self.device = device
            
        self.encoder = None
        self.decoder = None
        self.encoder_path = self.checkpoint_dir / "encoder.pth"
        self.decoder_path = self.checkpoint_dir / "decoder.pth"
        
        input_dims = (self.window_size, self.n_features)
        self._AuroEncoder(input_dims)
        
        self.to_device()
        self.history = {'train_loss': [], 'val_loss': []}
        self.scaler = GradScaler('cuda')
        
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
        # if data_augmentation:
        #     windows = self.augmenter.transform(windows)
        
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
        
        criterion = CombinedLoss(alpha=0.8,    # Bilanciamento tra ricostruzione e temporale
                                 beta=0.1,    # Peso per correlazioni tra features
                                 gamma=0.1     # Enfasi su eventi rari
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
                        # Forward pass - Handle skip connections
                        embedding, skip_connections = self.encoder.encode(batch)
                        reconstruction = self.decoder.decode(embedding, skip_connections)

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
                    # Forward pass - Handle skip connections
                    embedding, skip_connections = self.encoder.encode(batch)
                    reconstruction = self.decoder.decode(embedding, skip_connections)

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
                embedding, skip_connections = self.encoder.encode(batch)
                reconstruction = self.decoder.decode(embedding, skip_connections)

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

    def _AuroEncoder(self, input_dims):
        """
        Initialize Autoencoder
        """

        self.encoder = Autoencoder(input_size=input_dims,
                                   embedding_dim=self.embedding_dim,
                                   device=self.device,
                                   num_layers=config.num_layers
                                   ).to(self.device)
        
        self.decoder = Autoencoder(input_size=input_dims,
                                   embedding_dim=self.embedding_dim,
                                   device=self.device,
                                   num_layers=config.num_layers
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
                    embedding, _ = self.encoder.encode(batch)
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
                    # For inverse transform, we don't have skip connections - Create empty skip connections or use a simplified decoder path
                    skip_connections = [None] * len(self.decoder.decoder_layers)
                    reconstruction = self.decoder.decode(batch, skip_connections)
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
                embedding, _ = self.encoder.encode(batch)
                    
            embeddings.append(embedding.cpu().numpy())
            
        return np.concatenate(embeddings, axis=0)



def model_summary(model, input_size=None):
    """
    Print a summary of the model architecture similar to Keras' model.summary()
    
    Args:
        model: PyTorch model
        input_size: Tuple of input dimensions (excluding batch size) if needed for forward pass
    """
    def get_layer_info(layer):
        # Get number of trainable parameters
        num_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        
        # Get output shape
        output_size = "?"  # Default if we can't determine
        
        return {
            'class_name': layer.__class__.__name__,
            'num_params': num_params,
            'output_shape': output_size
        }
    
    print("\nModel Architecture Summary")
    print("=" * 100)
    print(f"{'Layer (type)':<40} {'Output Shape':<20} {'Param #':<15} {'Connected to':<20}")
    print("=" * 100)
    
    total_params = 0
    trainable_params = 0
    
    # Iterate through named modules to get layer info
    for name, layer in model.named_modules():
        # Skip the root module and container modules
        if name == "" or isinstance(layer, (nn.Sequential, Autoencoder)):
            continue
            
        layer_info = get_layer_info(layer)
        
        # Calculate parameters
        params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        trainable_params += params
        total_params += sum(p.numel() for p in layer.parameters())
        
        # Format the layer name and type
        layer_name = f"{name} ({layer_info['class_name']})"
        
        # Get input connections
        connected_to = ""
        try:
            for param in layer.parameters():
                if hasattr(param, '_backward_hooks'):
                    connected_to = str([hook for hook in param._backward_hooks.keys()])
        except:
            pass
        
        # Print layer information
        print(f"{layer_name:<40} {layer_info['output_shape']:<20} {params:<15,d} {connected_to:<20}")
    
    print("=" * 100)
    print(f"\nTotal params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {total_params - trainable_params:,}")
    
    # Calculate model size
    model_size = total_params * 4 / (1024 * 1024)  # Size in MB (assuming float32)
    print(f"Model size (MB): {model_size:.2f}")
    
    if hasattr(model, 'device'):
        print(f"Model device: {model.device}")
    else:
        print(f"Model device: {next(model.parameters()).device}")

def print_embedder_summary(embedder):
    """
    Print summary for both encoder and decoder of the embedder
    """
    print("\n" + "="*40 + " ENCODER " + "="*40)
    model_summary(embedder.encoder)
    
    print("\n" + "="*40 + " DECODER " + "="*40)
    model_summary(embedder.decoder)
    
    # Print additional embedder information
    print("\nEmbedder Configuration:")
    print(f"Window Size: {embedder.window_size}")
    print(f"Number of Features: {embedder.n_features}")
    print(f"Embedding Dimension: {embedder.embedding_dim}")
    print(f"Device: {embedder.device}")
    print(f"Checkpoint Directory: {embedder.checkpoint_dir}")