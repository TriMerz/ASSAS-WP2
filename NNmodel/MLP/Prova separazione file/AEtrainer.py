#!/usr/bin/env python3

# Standard library imports
import os

# Third party imports
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm
import matplotlib.pyplot as plt
from loss import *
from pathlib import Path

from encoderConfig import *
from Preprocessor import *
from loss import *
from AEmodels import *



class AutoEncoderTrainer:
    """
    Function defined inside the class:
        - fit
        - _train_epoch
        - _validate_epoch
        - _prepare_data
        - _create_dataloader
        - _save_checkpoint
        - _load_checkpoint
        - visualize_embeddings
        - plot_training_history
        - checkpoint_exists
        - evaluate_reconstruction
        - transform
        - inverse_transform
        - validate_input_shape
        - _batch_transform
    """
    def __init__(self,
                n_features: int,
                checkpoint_dir: str,
                window_size: int,
                embedding_dim: int = 256,
                encoder_class=None,
                decoder_class=None,
                encoder_kwargs=None,
                decoder_kwargs=None,
                custom_loss=None,
                device: str = None):
        """
        Args:
            - encoder_class: Any encoder class
            - decoder_class: Any decoder class 
            - custom_loss: Optional custom loss function
            - n_features: Number of input features
            - checkpoint_dir: Directory for saving checkpoints
            - window_size: Size of the input window
            - embedding_dim: Dimension of the embedding space
            - encoder_class: Class for the encoder (must implement encode method)
            - decoder_class: Class for the decoder (must implement decode method)
            - encoder_kwargs: Additional kwargs for encoder initialization
            - decoder_kwargs: Additional kwargs for decoder initialization
            - device: Device to use for training ('cuda' or 'cpu')
        """
        self.n_features = n_features
        self.window_size = window_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.embedding_dim = embedding_dim
        self.custom_loss = custom_loss
        
        # Device selection
        if device is None:
            if torch.cuda.is_available():
                gpu_id = torch.cuda.current_device()
                torch.cuda.set_device(gpu_id)
                self.device = f'cuda:{gpu_id}'
            else:
                self.device = 'cpu'
        else:
            self.device = device
            
        # Initialize model paths
        self.encoder_path = self.checkpoint_dir / "encoder.pth"
        self.decoder_path = self.checkpoint_dir / "decoder.pth"
        
        # Set default architecture if none provided
        if encoder_class is None or decoder_class is None:
            raise ValueError("Both encoder_class and decoder_class must be provided")
            
        # Validate that classes implement required methods
        self._validate_model_interface(encoder_class, "encode")
        self._validate_model_interface(decoder_class, "decode")

        # Initialize models with proper parameters
        encoder_params = {
            'input_size': (self.window_size, self.n_features),
            'embedding_dim': self.embedding_dim,
            'device': self.device,
            **(encoder_kwargs or {})
        }

        decoder_params = {
            'input_size': (self.window_size, self.n_features),
            'embedding_dim': self.embedding_dim,
            'device': self.device,
            **(decoder_kwargs or {})
        }
        
        self.encoder = encoder_class(**encoder_params)
        self.decoder = decoder_class(**decoder_params)
        
        # Move models to device
        self.to_device()
        
        # Initialize training components
        self.history = {'train_loss': [], 'val_loss': []}
        if self.device.startswith('cuda'):
            self.scaler = torch.amp.GradScaler()
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _validate_model_interface(self, model_class, required_method: str):
        """Validate that model class implements required methods"""
        if not hasattr(model_class, required_method):
            raise ValueError(
                f"Model class {model_class.__name__} must implement "
                f"the {required_method} method"
            )
        
    def to_device(self):
        """Move models to device efficiently"""
        if self.encoder:
            self.encoder = self.encoder.to(self.device)
        if self.decoder:
            self.decoder = self.decoder.to(self.device)
    
    def fit(self,
            windows,
            epochs: int = 100,
            batch_size: int = 32,
            learning_rate: float = 1e-4,
            validation_split: float = 0.2,
            weight_decay: float = 1e-4,
            patience: int = 20,
            use_amp: bool = True):
        """
        Training function with architecture-agnostic implementation
        """
        self.validate_input_shape(windows)
        
        # Data preparation
        if not isinstance(windows, torch.Tensor):
            windows = torch.FloatTensor(windows)
        if len(windows.shape) == 2:
            windows = windows.unsqueeze(0)
            
        # Prepare data loaders
        train_data, val_data = self._prepare_data(windows, validation_split)
        train_loader = self._create_dataloader(train_data, batch_size, shuffle=True)
        val_loader = self._create_dataloader(val_data, batch_size, shuffle=False)
        
        # Initialize optimizer and scheduler
        all_parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.AdamW(all_parameters, lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=len(train_loader)
        )
        
        # Initialize loss function
        criterion = CombinedLoss(alpha=0.8, beta=0.1, gamma=0.1).to(self.device)
        
        # Training state
        best_val_loss = float('inf')
        patience_counter = 0
        self.history = {'train_loss': [], 'val_loss': []}
        
        print(f"\nStarting training for {epochs} epochs:")
        print(f"{'Epoch':>5} {'Train Loss':>12} {'Val Loss':>12} {'Best':>6}")
        print("-" * 40)
        
        for epoch in range(epochs):
            # Training phase
            train_loss = self._train_epoch(
                train_loader,
                optimizer,
                criterion,
                scheduler,
                use_amp
            )
            
            # Validation phase
            val_loss = self._validate_epoch(val_loader, criterion)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Check for improvement
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
        
        # Plot final training history
        self.plot_training_history()

    def _forward_pass(self, batch, conditions=None, return_all=False):
        """Simplified forward pass"""
        encoder_output = self.encoder(batch)
        
        if isinstance(encoder_output, tuple):
            embedding, skip_connections = encoder_output
        else:
            embedding = encoder_output
            skip_connections = None
            
        reconstruction = self.decoder(embedding, skip_connections)
        
        if return_all:
            return {
                'embedding': embedding,
                'reconstruction': reconstruction,
                'skip_connections': skip_connections
            }
        return reconstruction

    def _compute_loss(self, batch, output_dict, criterion):
        """
        Compute loss handling custom loss functions
        """
        if self.custom_loss:
            return self.custom_loss(
                original=batch,
                reconstruction=output_dict['reconstruction'],
                embedding=output_dict['embedding'],
                metadata=output_dict['metadata']
            )
        return criterion(output_dict['reconstruction'], batch)

    def _train_epoch(self, train_loader, optimizer, criterion, scheduler, use_amp):
        """Enhanced training epoch handling all cases"""
        self.encoder.train()
        self.decoder.train()
        total_loss = 0
        
        # Reset states if needed
        if self.supports_state:
            if hasattr(self.encoder, 'reset_state'):
                self.encoder.reset_state()
            if hasattr(self.decoder, 'reset_state'):
                self.decoder.reset_state()
        
        for batch_idx, batch_data in enumerate(tqdm(train_loader, desc="Training")):
            # Handle different batch formats
            batch = batch_data[0] if isinstance(batch_data, (tuple, list)) else batch_data
            conditions = batch_data[1] if isinstance(batch_data, (tuple, list)) and len(batch_data) > 1 else None
            
            batch = batch.to(self.device)
            if conditions is not None:
                conditions = conditions.to(self.device)
            
            optimizer.zero_grad()
            
            try:
                if use_amp and torch.cuda.is_available():
                    with torch.amp.autocast(device_type='cuda'):
                        output_dict = self._forward_pass(batch, conditions, return_all=True)
                        loss = self._compute_loss(batch, output_dict, criterion)
                    
                    self.scaler.scale(loss).backward()
                    
                    # Handle gradient clipping if needed
                    if hasattr(self.encoder, 'clip_gradients'):
                        self.scaler.unscale_(optimizer)
                        self.encoder.clip_gradients()
                    if hasattr(self.decoder, 'clip_gradients'):
                        self.scaler.unscale_(optimizer)
                        self.decoder.clip_gradients()
                        
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    output_dict = self._forward_pass(batch, conditions, return_all=True)
                    loss = self._compute_loss(batch, output_dict, criterion)
                    loss.backward()
                    
                    # Handle gradient clipping if needed
                    if hasattr(self.encoder, 'clip_gradients'):
                        self.encoder.clip_gradients()
                    if hasattr(self.decoder, 'clip_gradients'):
                        self.decoder.clip_gradients()
                        
                    optimizer.step()
                
                if scheduler is not None:
                    scheduler.step()
                    
                total_loss += loss.item()
                
            except RuntimeError as e:
                print(f"\nError in batch {batch_idx + 1}:")
                print(f"Input shape: {batch.shape}")
                print(f"Device: {self.device}")
                print(f"Error: {str(e)}")
                raise e
                
        return total_loss / len(train_loader)

    @torch.no_grad()
    def _validate_epoch(self, val_loader, criterion):
        """Architecture-agnostic validation epoch"""
        self.encoder.eval()
        self.decoder.eval()
        total_loss = 0
        
        for batch in val_loader:
            if isinstance(batch, (tuple, list)):
                batch, *extra = batch
                conditions = extra[0] if extra else None
            else:
                conditions = None
                
            batch = batch.to(self.device)
            
            # Usa _forward_pass come nel train
            output_dict = self._forward_pass(batch, conditions, return_all=True)
            loss = self._compute_loss(batch, output_dict, criterion)
            total_loss += loss.item()
            
        return total_loss / len(val_loader)

    def transform(self, windows, conditions=None):
        """Simplified transform method"""
        self.encoder.eval()
        
        if isinstance(windows, np.ndarray):
            windows = torch.FloatTensor(windows)
        if len(windows.shape) == 2:
            windows = windows.unsqueeze(0)
            
        batch_size = 1024
        embeddings = []
        skip_connections_list = []
        
        with torch.no_grad():
            for i in range(0, len(windows), batch_size):
                batch = windows[i:i + batch_size].to(self.device)
                encoder_output = self._forward_pass(batch, return_all=True)
                embeddings.append(encoder_output['embedding'].cpu())
                
                if encoder_output['skip_connections'] is not None:
                    skip_connections_list.append(encoder_output['skip_connections'])
        
        embeddings = torch.cat(embeddings, dim=0).numpy()
        
        if skip_connections_list:
            return embeddings, skip_connections_list
        return embeddings

    def inverse_transform(self, embeddings, conditions=None, metadata=None):
        """Enhanced inverse transform with support for conditional models and metadata"""
        self.decoder.eval()
        
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.FloatTensor(embeddings)
        if len(embeddings.shape) == 1:
            embeddings = embeddings.unsqueeze(0)
            
        if conditions is not None and isinstance(conditions, np.ndarray):
            conditions = torch.FloatTensor(conditions)
            
        batch_size = 1024
        reconstructions = []
        
        with torch.no_grad():
            try:
                for i in range(0, len(embeddings), batch_size):
                    batch_emb = embeddings[i:i + batch_size].to(self.device)
                    batch_conditions = None
                    if conditions is not None:
                        batch_conditions = conditions[i:i + batch_size].to(self.device)
                        
                    # Prepare decoder input
                    decoder_input = {
                        'embedding': batch_emb,
                        'conditions': batch_conditions
                    }
                    
                    # Add metadata if available
                    if metadata is not None:
                        batch_metadata = metadata[i:i + batch_size]
                        decoder_input['skip_connections'] = batch_metadata
                    
                    reconstruction = self.decoder(**decoder_input)
                    reconstructions.append(reconstruction.cpu())
                    
                return torch.cat(reconstructions, dim=0).numpy()
                
            except RuntimeError as e:
                print(f"\nError during inverse transform:")
                print(f"Input shape: {embeddings.shape}")
                print(f"Device: {self.device}")
                print(f"Error: {str(e)}")
                raise

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

    def visualize_embeddings(self, windows, labels=None):
        """
        Embedding visualization using t-SNE
        """
        # Get embeddings efficiently using batch processing
        result = self.transform(windows)
        embeddings = result[0] if isinstance(result, tuple) else result
        
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
        """
        Check if checkpoint exists with proper error handling
        """
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
        total_length = windows_reshaped.shape[0] + self.window_size - 1
        
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
        if name == "" or isinstance(layer, (nn.Sequential, Performance)):
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
    Args:
        embedder: NonLinearEmbedder instance
    """
    print("\n" + "="*40 + " ENCODER " + "="*40)
    model_summary(embedder.encoder)
    
    print("\n" + "="*40 + " DECODER " + "="*40)
    model_summary(embedder.decoder)
    
    # Print additional embedder information
    print("\nEmbedder Configuration:")
    print(f"Architecture: {'Default' if embedder.default else 'Performance'}")
    print(f"Window Size: {embedder.window_size}")
    print(f"Number of Features: {embedder.n_features}")
    print(f"Embedding Dimension: {embedder.embedding_dim}")
    print(f"Device: {embedder.device}")
    print(f"Checkpoint Directory: {embedder.checkpoint_dir}")