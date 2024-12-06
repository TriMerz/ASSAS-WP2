
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Any



class WindowGenerator(Dataset):
    """
    Class with methods to generate sliding windows from time series data
    defined functions:
        - create_windows
        - __len__
        - __getitem__
        - get_dataloader
        - _batch_transform
        - _validate_embeddings
    """
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

    def create_windows(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create windows using numpy operations
        """
        n_samples = (len(self.data) - self.window_size) // self.stride + 1
        n_features = self.data.shape[1]

        # Create sliding windows efficiently using numpy
        indices = np.arange(self.window_size)[None, :] + \
                np.arange(n_samples)[:, None] * self.stride
        X = self.data[indices]  # X mantiene la struttura 3D - X.shape (n_samples, window_size, n_features)

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
        """
        Get a single window and target
        """
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size
        
        X = torch.FloatTensor(self.data[start_idx:end_idx])
        y = torch.FloatTensor(self.data[end_idx]) if end_idx < len(self.data) else \
            torch.zeros(self.data.shape[1], dtype=torch.float32)
        
        return X, y
    
    def get_dataloader(self,
                      shuffle: bool = True) -> DataLoader:
        """
        Create PyTorch DataLoader for efficient batching
        """
        return DataLoader(self,
                        batch_size=self.batch_size,
                        shuffle=shuffle,
                        num_workers=4,
                        pin_memory=True)
    
    def _batch_transform(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """
        Transform data in batches to avoid memory issues
        """
        n_samples = len(X)
        embeddings = []
        
        for i in range(0, n_samples, batch_size):
            batch = X[i:i + batch_size]
            batch_embedding = self.embedder.transform(batch)
            embeddings.append(batch_embedding)
            
        return np.concatenate(embeddings, axis=0)
    
    def _validate_embeddings(self, X: np.ndarray) -> None:
        """
        Optimized embedding validation
        """
        # Take multiple samples for validation
        n_samples = min(5, len(X))
        sample_indices = np.random.choice(len(X), n_samples, replace=False)
        
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