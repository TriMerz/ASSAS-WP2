#!/usr/bin/env python3

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = False
torch._dynamo.reset()
# Use eager mode instead
torch.backends.cudnn.benchmark = True

# All'inizio di AEmain.py
import gc
gc.collect()
torch.cuda.empty_cache()

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
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Third party imports
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from typing import Tuple, Optional, Any

# File imports
from AEtraining import *
from AEconfig import *
from HDFReader import *
from Preprocessor import *
from WindowGenerator import *



def copy_source_files(test_dir: str, new_test: bool):
    """
    Copy source files to test directory if this is a new test
    """
    if not new_test:
        return
        
    # Lista dei file da copiare
    files_to_copy = ['AEconfig.py', 'AEmain.py', 'AEmodels.py', 'AEtraining.py', 'config.yaml',
                     'HDFReader.py', 'loss.py', 'Preprocessor.py', 'WindowGenerator.py']
    
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
              f"\n- Using {'MACRO' if config.is_macro else 'MICRO'} data"
              f"\n- New test: {config.new_test}"
              f"\n- Test directory: {test_dir}"
              f"\n"
              f"\n- Scaler method: {config.scaler_method}"
              f"\n- num_layers: {config.num_layers}"
              f"\n- Window size: {config.window_size}"
              f"\n- Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

        print("\n2. Initializing HDF5 reader...")
        reader = HDF5Reader(config.database_path)
        
        print("\n3. Loading dataset...")
        if config.is_macro:
            df = reader.get_macro()
        else:
            df = reader.get_micro()
        print(f"Dataset loaded with shape: {df.shape}")
        print(df.head(5))
        
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
        X, y, T = window_gen.create_windows()
        print(f"Windows created with shapes:")
        print(f"- X: {X.shape}")
        print(f"- y: {y.shape}")
        print(f"- T: {T.shape}")
            
        print("\n7. Initializing embedder...")
        embedder = NonLinearEmbedder(checkpoint_dir=config.checkpoint_dir,
                                     n_features=X.shape[2],
                                     window_size=config.window_size,
                                     embedding_dim=config.embedding_dim,
                                     device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                     num_layers=config.num_layers,
                                     n_heads=config.n_heads,
                                     dropout=config.dropout_rate,
                                     )
    
        print("\n8. Preparing for embeddings...")
        X_tensor = torch.FloatTensor(X)
        print(f"Input tensor shape: {X_tensor.shape}")
        print(f"Device being used: {embedder.device}")
        
        print("\n8b. Printing model architecture...")
        print_embedder_summary(embedder)

        if config.new_test:
            print("\n9. Starting embedder training...")
            embedder.fit(X_tensor,
                         epochs=config.epochs,
                         batch_size=config.batch_size,
                         learning_rate=config.learning_rate,
                         validation_split=config.validation_split,
                         weight_decay=config.weight_decay,
                         patience=config.patience,
                         use_amp=True
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
