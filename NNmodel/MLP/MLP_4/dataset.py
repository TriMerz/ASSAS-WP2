#!/usr/bin/env python3

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = False
torch._dynamo.reset()
# Use eager mode instead
torch.backends.cudnn.benchmark = True

# Standard library imports
import os
import sys
import shutil
# Disable inductor and other warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TORCH_LOGS'] = ""
os.environ['TORCHDYNAMO_VERBOSE'] = "0"
os.environ['TORCH_INDUCTOR_VERBOSE'] = "0"
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCH_INDUCTOR_DISABLE'] = '1'

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
    def __init__(self, database, method, scalerpath):
        self.database = database
        self.method = method
        self.scalerpath = scalerpath
        self.index = database.index
        self.features_scaler = None
        
        # Separa timestamp e features
        self.timestamp = database['timestamp']
        self.features = database.drop('timestamp', axis=1)
    
    def encode_time(self, time_data):
        """
        Encode time features:
        1. Relative time from start
        2. Normalized time deltas
        """
        # Ensure time_data is numpy array
        if isinstance(time_data, pd.Series):
            time_data = time_data.values
            
        # 1. Relative time (normalized by sequence length)
        relative_time = (time_data - time_data[0]) / (time_data[-1] - time_data[0])
        
        # 2. Time deltas (normalized by mean delta)
        delta_time = np.diff(time_data, prepend=time_data[0])
        delta_time[0] = delta_time[1]  # Fix first value
        normalized_delta = delta_time / delta_time.mean()
        
        # Stack features
        time_features = np.column_stack([relative_time, normalized_delta])
        return time_features
    
    def __clean__(self):
        """Clean the DataFrame by removing NaNs and Duplicates."""
        df = self.database.dropna()                
        print(f"Removed {len(self.database) - len(df)} NaN values")
        df = df.drop_duplicates()       
        print(f"Removed {len(self.database) - len(df)} duplicates")
        self.database = df
        return self

    def __transform__(self):
        """Transform data with separate handling for time and other features."""
        if not os.path.exists(self.scalerpath):
            os.makedirs(self.scalerpath)
            
        scaler_filename = f"{self.method}Scaler.pkl"
        scaler_path = os.path.join(self.scalerpath, scaler_filename)
        
        # Handle non-time features
        if os.path.exists(scaler_path):
            self.features_scaler = joblib.load(scaler_path)
            print(f"{self.method}Scaler loaded successfully")
        else:
            if self.method == "Robust":
                self.features_scaler = RobustScaler()
            elif self.method == "MinMax":
                self.features_scaler = MinMaxScaler()
            else:
                self.features_scaler = StandardScaler()
            self.features_scaler.fit(self.features)
            joblib.dump(self.features_scaler, scaler_path)
            print(f"{self.method}Scaler created and saved")
        
        # Transform features
        scaled_features = pd.DataFrame(
            self.features_scaler.transform(self.features),
            columns=self.features.columns,
            index=self.index
        )
        
        # Encode time
        time_features = self.encode_time(self.timestamp)
        time_columns = ['relative_time', 'delta_time']
        encoded_time = pd.DataFrame(
            time_features,
            columns=time_columns,
            index=self.index
        )
        
        # Combine all features
        result = pd.concat([encoded_time, scaled_features], axis=1)
        print("\nTransformed data summary:")
        print(f"Time features shape: {encoded_time.shape}")
        print(f"Scaled features shape: {scaled_features.shape}")
        print(f"Final shape: {result.shape}")
        
        return result
    
    def __invTransform__(self, scaled_data):
        """
        Inverse transform scaled data, handling time features separately.
        """
        if self.features_scaler is None:
            raise ValueError("Scaler not found. Run transform() first.")

        # Convert input to 2D array if needed
        if len(scaled_data.shape) == 1:
            scaled_data = scaled_data.reshape(1, -1)

        # For embedded data
        if scaled_data.shape[1] == 256:  # Embedding dimension
            return pd.DataFrame(
                self.features_scaler.inverse_transform(scaled_data),
                columns=self.features.columns
            )
        
        # For regular data
        # Separate time features and regular features
        time_features = scaled_data[:, :2]  # First two columns are time features
        other_features = scaled_data[:, 2:]
        
        # Inverse transform only the non-time features
        original_features = pd.DataFrame(
            self.features_scaler.inverse_transform(other_features),
            columns=self.features.columns
        )
        
        # Reconstruct original timestamp if needed
        # Note: This is approximate since we normalized time
        start_time = self.timestamp.iloc[0]
        end_time = self.timestamp.iloc[-1]
        reconstructed_time = start_time + time_features[:, 0] * (end_time - start_time)
        
        time_df = pd.DataFrame(reconstructed_time, columns=['timestamp'])
        
        return pd.concat([time_df, original_features], axis=1)

    def process(self):
        """Execute the complete preprocessing pipeline."""
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
        X = self.data[indices]  # X mantiene la struttura 3D - X.shape (n_samples, window_size, n_features)
        """
            Non serve più transporre per il default encoder
            if not default:
                X = np.transpose(X, (0, 2, 1))
        """
        y = np.zeros((n_samples, n_features), dtype=np.float32)
        valid_indices = indices[:, -1] + 1 < len(self.data)
        y[valid_indices] = self.data[indices[valid_indices, -1] + 1]
        
        T = np.array([self.index[i:i + self.window_size + 1]
                    for i in range(0, len(self.index) - self.window_size, self.stride)])
        
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
            # Initialize embedder if not exists
            if self.embedder is None:
                self.embedder = NonLinearEmbedder(
                    n_features=self.input.shape[2],  # Ora prendiamo direttamente n_features
                    checkpoint_dir=checkpoint_dir,
                    window_size=self.input.shape[1],
                    default=True,
                    embedding_dim=256)
            
            # Validate input shape
            try:
                self.embedder.validate_input_shape(self.input)
                print(f"Valid input shape: {self.input.shape}")
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


def copy_source_files(test_dir: str, new_test: bool):
    """Copy source files to test directory if this is a new test"""
    if not new_test:
        return
        
    # Lista dei file da copiare
    files_to_copy = ['utils.py', 'loss.py', 'dataset.py', 'config.yaml', 'encoderConfig.py']
    
    # Ottieni il percorso della directory corrente
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    for file in files_to_copy:
        src = os.path.join(current_dir, file)
        dst = os.path.join(test_dir, file)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied {file} to test directory")

def debug_main():
    try:
        print("1. Setting up configuration...")
        config = setup_config()
        
        # Create test directory ONLY HERE, not in setup_config
        current_path = os.path.dirname(os.path.abspath(__file__))
        if config.new_test:
            test_dir = create_test_directory(config.model_name, current_path)
            # Copy source files only for new tests
            copy_source_files(test_dir, config.new_test)
        else:
            test_dir = os.path.join(current_path, f"{config.model_name}_{config.test_number}")
            if not os.path.exists(test_dir):
                raise ValueError(f"Test directory {test_dir} not found. Cannot load existing test.")
            
            # Verify checkpoints exist
            embedder_path = os.path.join(test_dir, "Embeddings")
            if not os.path.exists(os.path.join(embedder_path, "encoder.pth")) or \
               not os.path.exists(os.path.join(embedder_path, "decoder.pth")):
                raise ValueError(f"Embedder checkpoints not found in {embedder_path}")
        
        # Update config paths
        config.scalerpath = os.path.join(test_dir, "Scaler")
        config.checkpoint_dir = os.path.join(test_dir, "Embeddings")
        
        # Create necessary subdirectories
        os.makedirs(config.scalerpath, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        print(f"\nConfiguration loaded successfully:"
              f"\n- Database path: {config.database_path}"
              f"\n- Model: {config.model_name}"
              f"\n- Window size: {config.window_size}"
              f"\n- Using {'MACRO' if config.is_macro else 'MICRO'} data"
              f"\n- Embeddings enabled: {config.use_embeddings}"
              f"\n- New test: {config.new_test}"
              f"\n- Test directory: {test_dir}"
              f"\n- Device: {'cuda' if torch.cuda.is_available() else 'cpu'}"
              f"\n- layer_dim: {config.layer_dim}"
              f"\n- num_layers: {config.num_layers}")
        
        print("\n2. Initializing HDF5 reader...")
        reader = HDF5Reader(config.database_path)
        
        print("\n3. Loading dataset...")
        if config.is_macro:
            df = reader.get_macro()
        else:
            df = reader.get_micro()
        print(f"Dataset loaded with shape: {df.shape}")
        
        if df.empty:
            raise ValueError("Loaded DataFrame is empty")
        
        print("\n4. Preprocessing data...")
        preprocessor = DataPreprocessor(database=df, 
                                        method=config.scaler_method,
                                        scalerpath=config.scalerpath)
        
        scaled_data = preprocessor.process()
        print(f"Preprocessing complete. Scaled data shape: {scaled_data.shape}")
        print(f"\nScaled data ranges:")
        print(scaled_data.describe())
        
        print("\n5. Generating windows...")
        window_gen = WindowGenerator(preprocessed_data=scaled_data.values,
                                   original_index=scaled_data.index,
                                   scaler=preprocessor.features_scaler,
                                   window_size=config.window_size,
                                   stride=1,
                                   batch_size=config.batch_size)
        
        print("\n6. Creating window arrays...")
        X, y, T = window_gen.create_windows(default=config.default_encoder)
        print(f"Windows created with shapes:")
        print(f"- X: {X.shape}")
        print(f"- y: {y.shape}")
        print(f"- T: {T.shape}")

        if not config.use_embeddings:
            print("\nEmbeddings disabled in config. Exiting...")
            return
            
        print("\n7. Initializing embedder...")
        embedder = NonLinearEmbedder(n_features=X.shape[2],
                                    checkpoint_dir=config.checkpoint_dir,
                                    window_size=config.window_size,
                                    default=config.default_encoder,
                                    embedding_dim=config.embedding_dim,
                                    device='cuda' if torch.cuda.is_available() else 'cpu')
        
        print("\n8. Preparing for embeddings...")
        X_tensor = torch.FloatTensor(X)
        print(f"Input tensor shape: {X_tensor.shape}")
        print(f"Device being used: {embedder.device}")
        
        if config.new_test:
            print("\n9. Starting embedder training...")
            embedder.fit(X_tensor,
                         epochs=config.epochs,
                         batch_size=config.batch_size,
                         learning_rate=config.learning_rate,
                         validation_split=config.validation_split,
                         weight_decay=config.weight_decay,
                         patience=config.patience,
                         data_augmentation=config.data_augmentation
                         )
            print("\nTraining completed successfully!")
        else:
            print("\n9. Loading pre-trained embedder...")
            embedder._load_checkpoint()
            print("Embedder loaded successfully!")
        
        print("\n10. Creating embeddings...")
        X_embedded = embedder.transform(X)
        print(f"Final embedding shape: {X_embedded.shape}")
        
        print("\n11. Evaluating reconstruction quality...")
        embedder.evaluate_reconstruction(X)
        
        print("\n12. Visualizing embeddings...")
        embedder.visualize_embeddings(X)
        
    except Exception as e:
        print(f"\nError occurred during execution:")
        print(f"{'='*50}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"{'='*50}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_main()
