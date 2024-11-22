#!/usr/bin/env python3

# Standard library imports
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Third party imports
import pandas as pd
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import joblib
import matplotlib.pyplot as plt
import joblib
from typing import Tuple, Optional, Any

# File imports
from utils import *
from encoderConfig import *


class HDF5Reader(Dataset):
    def __init__(self,
                database_path,
                time=False):

        self.path = database_path
        self.current_file = None
        self.available_files = []
        self.current_file_index = 0
        self._scan_directory()

    def _scan_directory(self):
        """
        Scan directory for all .h5 files
        """
        if os.path.isdir(self.path):
            self.available_files = [
                os.path.join(self.path, f)
                for f in os.listdir(self.path)
                if f.endswith('.h5')
                ]
            if not self.available_files:
                raise FileNotFoundError('No .h5 files found in the specified directory')
            self.available_files.sort()  # Ensure consistent ordering
            self.current_file = self.available_files[0]
        else:
            if not self.path.endswith('.h5'):
                raise ValueError('Single file path must be an .h5 file')
            self.available_files = [self.path]
            self.current_file = self.path

    def next_dataset(self):
        """
        Move to the next dataset in the directory
        """
        if self.current_file_index < len(self.available_files) - 1:
            self.current_file_index += 1
            self.current_file = self.available_files[self.current_file_index]
            return True
        return False

    def get_current_filename(self):
        """
        Get the name of the current file being processed
        """
        return os.path.basename(self.current_file)

    def get_remaining_files(self):
        """
        Get number of remaining files to process
        """
        return len(self.available_files) - self.current_file_index - 1

    def get_total_files(self):
        """
        Get total number of .h5 files
        """
        return len(self.available_files)

    def get_micro(self):
        """
        Converts current H5 file's micro data into a pandas DataFrame.
        The varprim variables are expanded into separate columns.
        Time (microend) is included as the first column.
        
        Returns:
            pd.DataFrame: DataFrame containing the micro time-step data
        """
        with h5py.File(self.current_file, 'r') as f:
            micro_group = f['MACRO']['MICRO']
            varprim_matrix = micro_group['VARPRIM']['varprim'][:]
            microend = micro_group['microend'][:]
            
            df = pd.DataFrame(varprim_matrix, index=microend)
            df.insert(0, 'timestamp', microend)     # Add time as the first column
            
            print(f'File {self.get_current_filename()} has been read and converted into a Pandas DataFrame')
            return df

    def get_macro(self):
        """
        Converts current H5 file's macro data into a pandas DataFrame.
        Time (macroend) is included as the first column.
        
        Returns:
            pd.DataFrame: DataFrame containing the macro time-step data
        """
        with h5py.File(self.current_file, 'r') as f:
            OnlyMACRO_group = f['OnlyMACRO']
            varprim_matrix = OnlyMACRO_group['MACROvarprim'][:]
            macroend = OnlyMACRO_group['macroend'][:]
            
            df = pd.DataFrame(varprim_matrix, index=macroend)
            df.insert(0, 'timestamp', macroend)     # Add time as the first column
            
            print(f'File {self.get_current_filename()} has been read and converted into a Pandas DataFrame')
            return df

    @staticmethod
    def shape(df):
        return df.shape


class DataPreprocessor:
    """
    Preprocessing class for the raw data:
    - Remove NaNs and duplicates
    - Scale the data using MinMax or Standard scaler (scaler saved to - scalerpath -)

    Args:
        - database: DataFrame containing the raw data
        - method: string, either "MinMax", "Standard" or "Robust"
        - scalerpath: path to save/load the scaler
    """
    def __init__(self,
                 database,
                 method,
                 scalerpath):
        
        self.database = database
        self.method = method
        self.scalerpath = scalerpath
        self.index = database.index
        self.scaler = None

        self.timestamp = database['timestamp']     # Separate timestamp and features
        self.features = database.drop('timestamp', axis=1)
    
    def __clean__(self):
        """
        Clean the DataFrame by removing NaNs and Duplicates.
        """
        df = self.database.dropna()                
        print(f"Removed {len(self.database) - len(df)} NaN values")
        df = df.drop_duplicates()       
        print(f"Removed {len(self.database) - len(df)} duplicates")
        self.database = df
        return self

    def __transform__(self):
        """
        Scale the features (excluding timestamp) using the specified scaler.
            Returns:
                - DataFrame with original timestamp and scaled features.
        """
        if not os.path.exists(self.scalerpath):
            os.makedirs(self.scalerpath)
            
        scaler_filename = f"{self.method}Scaler.pkl"
        scaler_path = os.path.join(self.scalerpath, scaler_filename)
        
        # Load or create scaler
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"{self.method}Scaler loaded successfully")
        else:
            if self.method == "Robust":
                self.scaler = RobustScaler()
            elif self.method == "MinMax":
                self.scaler = MinMaxScaler()
            else:
                self.scaler = StandardScaler()
            self.scaler.fit(self.features)
            joblib.dump(self.scaler, scaler_path)
            print(f"{self.method}Scaler created and saved")
        
        scaled_features = pd.DataFrame(     # Scale features
            self.scaler.transform(self.features),
            columns=self.features.columns,
            index=self.index)
        
        result = pd.concat([self.timestamp, scaled_features], axis=1)     # Combine timestamp with scaled features
        return(result)
    
    def __invTransform__(self, scaled_data):
        """
        Inverse transform the scaled features (excluding timestamp).
        Args:
            - scaled_data: numpy array of scaled values (can be 1D or 2D)
        Returns:
            - DataFrame with original timestamp and unscaled features
        """
        if self.scaler is None:
            raise ValueError("Scaler not found. Run transform() first.")

        # Convert input to 2D array if it's 1D
        if len(scaled_data.shape) == 1:
            scaled_data = scaled_data.reshape(1, -1)

        # For embedded data, we need to handle differently since we don't have timestamp
        if scaled_data.shape[1] == 256:  # If this is embedded data
            original_features = pd.DataFrame(
                self.scaler.inverse_transform(scaled_data),
                columns=self.features.columns)
            # Use the first timestamp as placeholder or create a new index
            result = pd.concat([pd.DataFrame(self.timestamp.iloc[0:1]), original_features], axis=1)
            
        else:  # Original data with timestamp
            timestamp = pd.DataFrame(scaled_data[:, 0])     # Separate timestamp and features
            features = pd.DataFrame(scaled_data[:, 1:])
            
            original_features = pd.DataFrame(     # Inverse transform features
                self.scaler.inverse_transform(features),
                columns=features.columns,
                index=features.index)
            
            result = pd.concat([timestamp, original_features], axis=1)     # Combine timestamp with unscaled features
        
        return result

    def process(self):
        """
        Execute the complete preprocessing pipeline.
        """
        return self.__clean__().__transform__()


class WindowGenerator(Dataset):
    def __init__(self,
                 preprocessed_data,  # può essere DataFrame o ndarray
                 original_index,
                 scaler: Any,
                 window_size: int,
                 stride: int = 1,
                 batch_size: int = 32):
        
        # Converti DataFrame in numpy array se necessario
        if isinstance(preprocessed_data, pd.DataFrame):
            self.data = preprocessed_data.values
        else:
            self.data = preprocessed_data
            
        self.data = self.data.astype(np.float32)  # Assicura tipo corretto
        self.index = original_index
        self.scaler = scaler
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size
        self.n_windows = (len(self.data) - window_size) // stride + 1

    def create_windows(self, default: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create windows using numpy operations"""
        n_samples = (len(self.data) - self.window_size) // self.stride + 1
        n_features = self.data.shape[1]
        
        # Create sliding windows efficiently using numpy
        indices = np.arange(self.window_size)[None, :] + \
                 np.arange(n_samples)[:, None] * self.stride
        
        # Create X and y
        X = self.data[indices]
        y = np.zeros((n_samples, n_features), dtype=np.float32)
        valid_indices = indices[:, -1] + 1 < len(self.data)
        y[valid_indices] = self.data[indices[valid_indices, -1] + 1]
        
        # Create time windows
        T = np.array([self.index[i:i + self.window_size + 1]
                     for i in range(0, len(self.index) - self.window_size, self.stride)])
        
        if not default:
            X = np.transpose(X, (0, 2, 1))
        
        print(f"Created windows with shapes: X: {X.shape}, y: {y.shape}, T: {T.shape}")
        return X, y, T

    def __len__(self) -> int:
        return self.n_windows
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single window and target"""
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size
        
        X = torch.FloatTensor(self.data[start_idx:end_idx])
        y = torch.FloatTensor(self.data[end_idx]) if end_idx < len(self.data) else \
            torch.zeros(self.data.shape[1], dtype=torch.float32)
        
        return X, y
    
    def get_dataloader(self,
                      shuffle: bool = True) -> DataLoader:
        """Create PyTorch DataLoader for efficient batching"""
        return DataLoader(self,
                        batch_size=self.batch_size,
                        shuffle=shuffle,
                        num_workers=4,
                        pin_memory=True)

    def create_embeddings(self,
                         X: np.ndarray,
                         checkpoint_dir: Optional[str] = None,
                         test: bool = False,
                         epochs: int = 100,
                         batch_size: int = 32,
                         learning_rate: float = 0.001,
                         validation_split: float = 0.2,
                         weight_decay: float = 0.0001,
                         patience: int = 15,
                         data_augmentation: bool = False) -> np.ndarray:
        """Optimized version of create_embeddings with better memory management and faster training"""
        
        self.input = X
        self.epoch = epochs
        self.batch_size = batch_size
        
        if X is None:
            raise ValueError("No input data provided for embedding creation")
        
        if self.default:
            # Reshape X efficiently using views when possible
            try:
                self.reshaped_X = self.input.reshape(self.input.shape[0], -1)
            except ValueError:
                self.reshaped_X = self.input.copy().reshape(self.input.shape[0], -1)
            
            # Initialize embedder if not exists
            if self.embedder is None:
                self.embedder = NonLinearEmbedder(
                    n_features=self.input.shape[2],
                    checkpoint_dir=checkpoint_dir,
                    window_size=self.input.shape[1],
                    default=True,
                    embedding_dim=256
                )
            
            # Validate input shape
            try:
                self.embedder.validate_input_shape(self.reshaped_X)
                print(f"Valid input shape: {self.reshaped_X.shape}")
            except ValueError as e:
                print(f"Invalid input shape: {e}")
                sys.exit(1)
            
            # Load or train embedder with optimized training
            if self.embedder.checkpoint_exists():
                print("Loading existing checkpoint...")
                self.embedder.load_checkpoint()
            else:
                # Create DataLoader for efficient batching
                train_data = torch.FloatTensor(self.reshaped_X)
                train_dataset = TensorDataset(train_data)
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True
                )
                
                # Train with optimizations
                self.embedder.fit(
                    train_loader,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    validation_split=validation_split,
                    weight_decay=weight_decay,
                    patience=patience,
                    data_augmentation=data_augmentation,
                    use_amp=True  # Enable automatic mixed precision
                )
            
            # Optional embedding validation
            if test:
                self._validate_embeddings(self.input)
                
            # Transform in batches for memory efficiency
            return self._batch_transform(self.reshaped_X)
            
        else:
            # Performance encoder case
            if self.embedder is None:
                self.embedder = NonLinearEmbedder(
                    n_features=self.input.shape[1],
                    checkpoint_dir=checkpoint_dir,
                    window_size=self.input.shape[2],
                    default=False,
                    embedding_dim=256
                )
            
            # Rest of the performance encoder logic...
            # Similar optimizations as above
            return self.embedder.transform(self.input)
    
    def _batch_transform(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """Transform data in batches to avoid memory issues"""
        n_samples = len(X)
        embeddings = []
        
        for i in range(0, n_samples, batch_size):
            batch = X[i:i + batch_size]
            batch_embedding = self.embedder.transform(batch)
            embeddings.append(batch_embedding)
            
        return np.concatenate(embeddings, axis=0)
    
    def _validate_embeddings(self, X: np.ndarray) -> None:
        """Optimized embedding validation"""
        # Take multiple samples for validation
        n_samples = min(5, len(X))
        sample_indices = np.random.choice(len(X), n_samples, replace=False)
        
        if self.default:
            sample_windows = X[sample_indices]
            full_validation_input = self.reshaped_X
        else:
            sample_windows = X[sample_indices]
            full_validation_input = X
        
        # Convert to torch tensor and ensure proper device
        sample_windows_tensor = torch.from_numpy(sample_windows).float().to(self.embedder.device)
        
        # Process in batches
        embeddings = self._batch_transform(sample_windows_tensor)
        reconstructed = self.embedder.inverse_transform(embeddings)
        
        # Calculate reconstruction error
        with torch.no_grad():
            reconstruction_error = np.mean((sample_windows.flatten() - reconstructed.flatten())**2)
        
        print(f"-------------------------------------------------")
        print(f"Sample reconstruction error: {reconstruction_error:.6f}")
        print(f"Input shape: {sample_windows.shape}")
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Reconstructed shape: {reconstructed.shape}\n")
        
        # Full validation
        self.embedder.evaluate_reconstruction(full_validation_input)
        self.embedder.visualize_embeddings(full_validation_input)


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
import tslearn.metrics as metrics
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
        # Calculate intermediate dimensions
        dim1 = input_size // 2
        dim2 = dim1 // 2
        dim3 = dim2 // 2
        dim4 = dim3 // 2

        self.encoder = nn.Sequential(
            nn.LayerNorm(input_size),
            ResidualBlock(input_size, dim1),
            nn.Dropout(0.1),
            ResidualBlock(dim1, dim2),
            nn.Dropout(0.1),
            ResidualBlock(dim2, dim3),
            nn.Dropout(0.1),
            ResidualBlock(dim3, dim4),
            nn.Dropout(0.1),
            ResidualBlock(dim4, embedding_dim)
        )

        self.decoder = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            ResidualBlock(embedding_dim, dim4),
            nn.Dropout(0.1),
            ResidualBlock(dim4, dim3),
            nn.Dropout(0.1),
            ResidualBlock(dim3, dim2),
            nn.Dropout(0.1),
            ResidualBlock(dim2, dim1),
            nn.Dropout(0.1),
            ResidualBlock(dim1, input_size)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        embedding = self.encode(x)
        reconstruction = self.decode(embedding)
        return embedding, reconstruction


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
        """
        Encode the input data
            - x.shape = (batch, n_features, window_size) and first feature is time
        """
        times = x[:, 0, :]  # (batch, window_size)
        features = x[:, 1:, :]  # (batch, n_features-1, window_size)
        
        # Encode time
        time_encoding = self.time_encoder(times)  # (batch, window_size, 64)
        time_encoding = time_encoding.transpose(1, 2)  # (batch, 64, window_size)
        
        # Concatenate time encoding with other features
        x_combined = torch.cat([time_encoding, features], dim=1)
        
        # Get embedding
        embedding = self.encoder(x_combined)
        return(embedding)

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
        self.checkpoint_dir = Path(checkpoint_dir)
        self.window_size = window_size
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
        self.scaler = GradScaler()  # For mixed precision training
        self.augmenter = TimeSeriesAugmenter()
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def to_device(self):
        """Move models to device efficiently"""
        if self.encoder:
            self.encoder = self.encoder.to(self.device)
        if self.decoder:
            self.decoder = self.decoder.to(self.device)
    
    @torch.cuda.amp.autocast()
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
        """
        Optimized training with automatic mixed precision and better memory management
        """
        # Data augmentation if requested
        if data_augmentation:
            windows = self.augmenter.transform(windows)
        
        # Prepare data loaders
        train_data, val_data = self._prepare_data(windows, validation_split)
        train_loader = self._create_dataloader(train_data, batch_size, shuffle=True)
        val_loader = self._create_dataloader(val_data, batch_size, shuffle=False)
        
        # Initialize optimizer and scheduler
        optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        scheduler = torch.optim.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=len(train_loader)
        )
        
        criterion = CombinedLoss(alpha=0.7).to(self.device)
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Compile models for faster training if torch 2.0+ is available
        if hasattr(torch, 'compile'):
            self.encoder = torch.compile(self.encoder)
            self.decoder = torch.compile(self.decoder)
        
        for epoch in range(epochs):
            # Training
            train_loss = self._train_epoch(
                train_loader, optimizer, criterion, scheduler, use_amp)
            
            # Validation
            val_loss = self._validate_epoch(val_loader, criterion)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint()
                print(f"Checkpoints saved | ", end="")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                self._load_checkpoint()
                break
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}")
            
    def _train_epoch(self, train_loader, optimizer, criterion, scheduler, use_amp):
        """Optimized training epoch with AMP support"""
        self.encoder.train()
        self.decoder.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    embeddings, reconstructed = self.encoder(batch)
                    loss = criterion(reconstructed, batch)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                embeddings, reconstructed = self.encoder(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    @torch.no_grad()
    def _validate_epoch(self, val_loader, criterion):
        """Optimized validation epoch"""
        self.encoder.eval()
        self.decoder.eval()
        total_loss = 0
        
        for batch in val_loader:
            embeddings, reconstructed = self.encoder(batch)
            loss = criterion(reconstructed, batch)
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
    
    def transform(self, windows):
        """Optimized transform with batch processing"""
        self.encoder.eval()
        
        # Handle different input types
        if isinstance(windows, np.ndarray):
            windows = torch.FloatTensor(windows)
        
        # Process in batches
        batch_size = 1024  # Adjust based on available memory
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(windows), batch_size):
                batch = windows[i:i + batch_size].to(self.device)
                embedding = self.encoder.encode(batch)
                embeddings.append(embedding.cpu())
        
        return torch.cat(embeddings, dim=0).numpy()
    
    def inverse_transform(self, embeddings):
        """Optimized inverse transform with batch processing"""
        self.decoder.eval()
        
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.FloatTensor(embeddings)
        
        batch_size = 1024
        reconstructions = []
        
        with torch.no_grad():
            for i in range(0, len(embeddings), batch_size):
                batch = embeddings[i:i + batch_size].to(self.device)
                reconstruction = self.decoder.decode(batch)
                reconstructions.append(reconstruction.cpu())
        
        return torch.cat(reconstructions, dim=0).numpy()
    
    def _save_checkpoint(self):
        """Efficient checkpoint saving"""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'history': self.history
        }, self.encoder_path)
    
    def _load_checkpoint(self):
        """Efficient checkpoint loading"""
        checkpoint = torch.load(self.encoder_path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.history = checkpoint['history']

    def _default_encoder(self, input_size):
        """
        Build optimized encoder architecture for high-dimensional non-linear data.
        """
        self.encoder = Default(input_size=input_size,
                            embedding_dim=self.embedding_dim).to(self.device)
        self.decoder = Default(input_size=self.embedding_dim,
                            embedding_dim=input_size).to(self.device)
        
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
        """
        Optimized training history visualization
        """
        plt.figure(figsize=(10, 6))
        
        # Use numpy for efficient plotting
        epochs = np.arange(1, len(self.history['train_loss']) + 1)
        train_losses = np.array(self.history['train_loss'])
        val_losses = np.array(self.history['val_loss'])
        
        plt.plot(epochs, train_losses, label='Training Loss')
        plt.plot(epochs, val_losses, label='Validation Loss')
        
        # Add moving average for smoothing
        window = min(5, len(epochs) // 10)
        if window > 1:
            train_ma = np.convolve(train_losses, np.ones(window)/window, mode='valid')
            val_ma = np.convolve(val_losses, np.ones(window)/window, mode='valid')
            plt.plot(epochs[window-1:], train_ma, 'k--', alpha=0.5, label='Training MA')
            plt.plot(epochs[window-1:], val_ma, 'r--', alpha=0.5, label='Validation MA')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot with high resolution
        plot_path = self.checkpoint_dir / 'training_history.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training History Plot saved at: {plot_path}")

    def checkpoint_exists(self):
        """Check if checkpoint exists with proper error handling"""
        try:
            return self.encoder_path.exists() and self.decoder_path.exists()
        except Exception as e:
            print(f"Error checking checkpoint existence: {e}")
            return False

    def evaluate_reconstruction(self, windows, n_samples=5):
        """
        Optimized reconstruction quality evaluation with batch processing
        """
        # Get embeddings and reconstruct efficiently
        emb = self.transform(windows)
        reconstructed = self.inverse_transform(emb)
        
        # Move computation to numpy for efficiency
        if self.default:
            # Reshape efficiently using views when possible
            try:
                windows_reshaped = windows.reshape(windows.shape[0], self.window_size, -1)
                reconstructed_reshaped = reconstructed.reshape(reconstructed.shape[0], self.window_size, -1)
            except ValueError:
                windows_reshaped = windows.copy().reshape(windows.shape[0], self.window_size, -1)
                reconstructed_reshaped = reconstructed.copy().reshape(reconstructed.shape[0], self.window_size, -1)
            
            n_features = windows_reshaped.shape[2]
            total_length = (windows_reshaped.shape[0] - 1) * self.window_size + 1
            
        else:
            n_features = windows.shape[1]
            total_length = (windows.shape[0] - 1) * self.window_size + 1
        
        # Preallocate arrays
        original_series = np.zeros((total_length, n_features))
        reconstructed_series = np.zeros((total_length, n_features))
        counts = np.zeros(total_length)
        
        # Optimize reconstruction using vectorized operations where possible
        if self.default:
            for i in range(windows_reshaped.shape[0]):
                start_idx = i
                end_idx = start_idx + self.window_size
                mask = end_idx <= total_length
                if mask:
                    idx_range = np.arange(start_idx, end_idx)
                    original_series[idx_range] += windows_reshaped[i]
                    reconstructed_series[idx_range] += reconstructed_reshaped[i]
                    counts[idx_range] += 1
        else:
            for i in range(windows.shape[0]):
                start_idx = i
                end_idx = start_idx + self.window_size
                mask = end_idx <= total_length
                if mask:
                    idx_range = np.arange(start_idx, end_idx)
                    original_series[idx_range] += windows[i].T
                    reconstructed_series[idx_range] += reconstructed[i].T
                    counts[idx_range] += 1
        
        # Vectorized mean calculation
        mask = counts > 0
        original_series[mask] /= counts[mask, np.newaxis]
        reconstructed_series[mask] /= counts[mask, np.newaxis]
        
        # Plotting and metrics calculation
        time = original_series[:, 0]
        feature_indices = np.random.choice(n_features-1, n_samples, replace=False) + 1
        
        fig, axes = plt.subplots(n_samples, 1, figsize=(15, 4*n_samples))
        if n_samples == 1:
            axes = [axes]
        
        metrics_dict = {}
        for i, feat_idx in enumerate(feature_indices):
            # Plot
            axes[i].plot(time, original_series[:, feat_idx],
                        label='Original', color='blue', alpha=0.7)
            axes[i].plot(time, reconstructed_series[:, feat_idx],
                        label='Reconstructed', color='red', alpha=0.7, linestyle='--')
            
            axes[i].set_title(f'Feature {feat_idx}')
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel('Value')
            axes[i].legend()
            axes[i].grid(True)
            
            # Calculate metrics efficiently
            diff = original_series[:, feat_idx] - reconstructed_series[:, feat_idx]
            metrics_dict[feat_idx] = {
                'MSE': np.mean(diff**2),
                'MAE': np.mean(np.abs(diff)),
                'Correlation': np.corrcoef(original_series[:, feat_idx],
                                        reconstructed_series[:, feat_idx])[0,1],
                'R2': metrics.r2_score(original_series[:, feat_idx],
                                    reconstructed_series[:, feat_idx])
            }
        
        plt.tight_layout()
        plot_path = self.checkpoint_dir / 'reconstruction_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Print metrics
        for feat_idx, metric_values in metrics_dict.items():
            print(f"\nFeature {feat_idx}:")
            for metric_name, value in metric_values.items():
                print(f"  {metric_name}: {value:.6f}")

    def validate_input_shape(self, windows):
        """
        Optimized input shape validation with better error messages
        """
        shape_error = False
        error_message = ""
        
        try:
            if not self.default:
                if len(windows.shape) == 2:
                    if (windows.shape[0] != self.n_features and 
                        windows.shape[1] != self.window_size):
                        if (windows.shape[0] == self.window_size and 
                            windows.shape[1] == self.n_features):
                            print("Warning: Window shape will be transposed")
                        else:
                            shape_error = True
                elif len(windows.shape) == 3:
                    if (windows.shape[1] != self.n_features or 
                        windows.shape[2] != self.window_size):
                        if (windows.shape[1] == self.window_size and 
                            windows.shape[2] == self.n_features):
                            print("Warning: Windows will be transposed")
                        else:
                            shape_error = True
            else:
                expected_size = self.window_size * self.n_features
                if len(windows.shape) == 1 and windows.shape[0] != expected_size:
                    shape_error = True
                elif len(windows.shape) == 2 and windows.shape[1] != expected_size:
                    shape_error = True
            
            if shape_error:
                raise ValueError(f"Invalid shape: {windows.shape}. "
                            f"Expected shape information has been logged.")
            
            return True
        
        except Exception as e:
            print(f"Error during shape validation: {e}")
            raise

#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

# Loss function for the autoencoder
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        
    def forward(self, reconstructed, original):
        # MSE loss
        mse_loss = self.mse(reconstructed, original)
        
        # Temporal consistency loss
        temp_loss = self.temporal_consistency(reconstructed, original)
        
        return self.alpha * mse_loss + (1 - self.alpha) * temp_loss
    
    def temporal_consistency(self, reconstructed, original):
        # Calculate differences between consecutive time steps
        rec_diff = reconstructed[:, 1:] - reconstructed[:, :-1]
        orig_diff = original[:, 1:] - original[:, :-1]
        
        return F.mse_loss(rec_diff, orig_diff)


# config.py
import argparse
import yaml
import os
from dataclasses import dataclass

@dataclass
class ModelConfig:
    # Required arguments (no default values) first
    database_path: str
    model_name: str
    window_size: int
    use_embeddings: bool
    is_macro: bool
    
    # Optional arguments (with default values) after
    scalerpath: str = ""
    checkpoint_dir: str = ""
    new_test: bool = False
    test_number: int = 0
    embedding_dim: int = 256
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    weight_decay: float = 0.0001
    patience: int = 15
    data_augmentation: bool = False
    scaler_method: str = "MinMax"
    default_encoder: bool = True
    time_encoding_dim: int = 64
    conv_channels: int = 128
    n_heads: int = 8
    dropout_rate: float = 0.1
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Create config from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

def parse_args():
    parser = argparse.ArgumentParser(description='Time Series Processing and Embedding')
    
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML configuration file')
    
    parser.add_argument('--database_path', type=str,
                        help='Path to HDF5 database directory or file')
    
    parser.add_argument('--model_name', type=str, 
                        choices=["CNN", "FNO", "GraFITi", "MLP", "RNN", "Transformer"],
                        help='Name of the model to use')
    
    parser.add_argument('--window_size', type=int,
                        help='Size of the sliding window')
    
    parser.add_argument('--use_embeddings', action='store_true',
                        help='Whether to use embeddings')
    
    parser.add_argument('--is_macro', action='store_true',
                        help='Use MACRO (True) or MICRO (False) data')
    
    parser.add_argument('--new_test', action='store_true',
                        help='Create new test directory')
    
    parser.add_argument('--test_number', type=int,
                        help='Specific test number to use (if not new_test)')
    
    parser.add_argument('--scaler_method', type=str,
                        choices=["MinMax", "Standard", "Robust"],
                        help='Method to use for scaling data')
    
    return parser.parse_args()

def create_test_directory(model_name: str, base_path: str, test_number: int = None) -> str:
    """Create a new test directory with incremental numbering"""
    if test_number is not None:
        test_dir = os.path.join(base_path, f"{model_name}_{test_number}")
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        return test_dir
    
    test_num = 0
    while True:
        test_dir = os.path.join(base_path, f"{model_name}_{test_num}")
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
            return test_dir
        test_num += 1

def setup_config():
    """Setup configuration from command line args and config file"""
    args = parse_args()
    
    # Load base config from YAML
    config = ModelConfig.from_yaml(args.config)
    
    # Override with command line arguments if provided
    for arg in vars(args):
        if getattr(args, arg) is not None:
            setattr(config, arg, getattr(args, arg))
    
    # Create directories
    current_path = os.path.dirname(os.path.abspath(__file__))
    
    if config.new_test:
        test_dir = create_test_directory(config.model_name, current_path)
    else:
        test_dir = create_test_directory(config.model_name, current_path, config.test_number)
    
    # Update paths
    config.scalerpath = os.path.join(test_dir, "Scaler")
    config.checkpoint_dir = os.path.join(test_dir, "Embeddings")
    
    # Create necessary directories
    os.makedirs(config.scalerpath, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    return config


# Main function
def main(config):
    try:
        reader = HDF5Reader(config.database_path)
        if config.is_macro:
            df = reader.get_macro()
        else:
            df = reader.get_micro() 
        print(f"-------------------------------------------------")
        print(f"Shape of the original DataFrame: {df.shape} \n")

        preprocessor = DataPreprocessor(df,
                                      method=config.scaler_method,
                                      scalerpath=config.scalerpath)
        scaled_data = preprocessor.process()
        print(f"-------------------------------------------------")
        print(f"Shape of the scaled DataFrame: {scaled_data.shape} \n")

        window_gen = WindowGenerator(
            preprocessed_data=scaled_data,
            original_index=df.index,
            scaler=preprocessor.scaler,
            window_size=config.window_size
        )

        # Create the windows
        X, y, T = window_gen.create_windows(default=config.default_encoder)
        print(f"-------------------------------------------------")
        print(f"Shape of X: {np.shape(X)}, Shape of y: {np.shape(y)}, Shape of T: {np.shape(T)} \n")

        if config.use_embeddings:
            print("Starting embedding training...")
            embedder = NonLinearEmbedder(
                n_features=X.shape[2],  # number of features
                checkpoint_dir=config.checkpoint_dir,
                window_size=config.window_size,
                default=config.default_encoder,
                embedding_dim=config.embedding_dim,
                device='cpu'  # since you don't have GPU
            )

            print("Training embedder...")
            embedder.fit(
                windows=X,
                epochs=config.epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                validation_split=config.validation_split,
                weight_decay=config.weight_decay,
                patience=config.patience,
                data_augmentation=config.data_augmentation,
                use_amp=False  # since you're on CPU
            )

            # After training, transform the data
            print("Creating embeddings...")
            X = embedder.transform(X)
            print(f"Final embedding shape: {X.shape}")

            # Evaluate reconstruction quality
            print("Evaluating reconstruction quality...")
            embedder.evaluate_reconstruction(X, n_samples=5)

            # Visualize embeddings
            print("Visualizing embeddings...")
            embedder.visualize_embeddings(X)

            print("Training completed!")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")



# config.yaml

# Test configuration
new_test: true  # Whether to create a new test directory
test_number: 4   # Specific test number to use (if not new_test)

# Data paths
database_path: "/data/savini/ASSAS/WP2/ASSAS-WP2/NNmodel/HDFdatasets"
scalerpath: ""  # Will be set automatically based on test directory
checkpoint_dir: ""  # Will be set automatically based on test directory

# Data processing
window_size: 10
use_embeddings: true
is_macro: true  # True for MACRO, False for MICRO

# Model parameters
model_name: "MLP" # Options: "CNN", "FNO", "GraFITi", "MLP", "RNN", "Transformer"
embedding_dim: 256  # Dimension of the embedded representation

# Training parameters
epochs: 100
batch_size: 32
learning_rate: 0.001
validation_split: 0.2
weight_decay: 0.0001
patience: 15
data_augmentation: false

# Preprocessing parameters
scaler_method: "MinMax"  # Options: "MinMax", "Standard", "Robust"

# WindowGenerator parameters
default_encoder: true  # True for default, False for performance encoder

# Time encoding parameters (for performance encoder)
time_encoding_dim: 64

# Neural network parameters
conv_channels: 128  # Number of channels in convolutional layers
n_heads: 8         # Number of attention heads in transformer blocks
dropout_rate: 0.1  # Dropout rate for neural networks

# Parameter descriptions:
# new_test: Whether to create a new test directory
# test_number: Specific test number to use if not creating new test
# database_path: Path to the directory containing HDF5 files or single HDF5 file
# scalerpath: Directory to save/load data scalers
# checkpoint_dir: Directory to save model checkpoints and embeddings
# window_size: Number of time steps in each sliding window
# use_embeddings: Whether to use dimensional reduction through embeddings
# is_macro: Use macro-scale (true) or micro-scale (false) data
# model_name: Name of the model architecture to use
# embedding_dim: Dimension of the embedded representation
# epochs: Number of training epochs
# batch_size: Number of samples per training batch
# learning_rate: Learning rate for optimizer
# validation_split: Fraction of data to use for validation
# weight_decay: L2 regularization parameter
# patience: Number of epochs without improvement before early stopping
# data_augmentation: Whether to use data augmentation during training
# scaler_method: Method to use for data scaling
# default_encoder: Use default or performance-oriented encoder architecture
# time_encoding_dim: Dimension of time encoding for performance encoder
# conv_channels: Number of channels in convolutional layers
# n_heads: Number of attention heads in transformer blocks
# dropout_rate: Dropout rate for neural networks



if __name__ == "__main__":
    config = setup_config()
    main(config)