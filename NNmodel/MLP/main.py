#!/usr/bin/env python3


import os
import time
import configparser
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
import time
from tqdm import tqdm



def setupArgs():
    parser = argparse.ArgumentParser()
    config = configparser.ConfigParser()

    parser.add_argument("--config",
                        type=str,
                        default=os.path.join(os.path.dirname(__file__), "config.ana"),
                        help="Path to the config file.")
    config_path = parser.parse_args().config

    if os.path.exists(config_path):
        config.read(config_path)
    else:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    return(config)






class TrainingManager:
    def __init__(self,
                 model,
                 optimizer,
                 criterion,
                 data_handler,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 batch_size=32,
                 num_epochs=10,
                 early_stopping_patience=5):
        
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.data_handler = data_handler
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        
        print(f"Training will run on {self.device}")
        self.model.to(self.device)

    # train_single_epoch is defined in training.py

    def train_all_epochs(self, current_dataset_name):
        """Esegue il training su tutte le epoche per un singolo dataset"""
        print(f"\nTraining on dataset: {current_dataset_name}")
        
        # Prepara il DataLoader per il dataset corrente
        # Nota: Qui dovresti adattare la creazione del DataLoader in base al tuo specifico caso
        train_data = self.prepare_dataset_for_training(self.data_handler.get_micro())
        train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            epoch_loss = self.train_single_epoch(train_loader, epoch)
            
            # Early stopping check
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
                # Salva il miglior modello
                self.save_checkpoint(f"best_model_{current_dataset_name}.pth")
            else:
                patience_counter += 1
                
            if patience_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
                
        return best_loss

    def train_all_datasets(self):
        """Esegue il training su tutti i dataset sequenzialmente"""
        overall_start_time = time.time()
        training_history = []
        
        dataset_count = self.data_handler.get_total_files()
        print(f"Starting training on {dataset_count} datasets")
        
        while True:
            current_file = self.data_handler.get_current_filename()
            start_time = time.time()
            
            # Training su tutte le epoche per il dataset corrente
            best_loss = self.train_all_epochs(current_file)
            
            # Registra i risultati
            training_history.append({
                'dataset': current_file,
                'best_loss': best_loss,
                'time': time.time() - start_time
            })
            
            # Passa al dataset successivo se disponibile
            if not self.data_handler.next_dataset():
                break
        
        # Stampa il riepilogo finale
        total_time = time.time() - overall_start_time
        self.print_training_summary(training_history, total_time)

    def prepare_dataset_for_training(self, df):
        """
        Converte il DataFrame in un formato adatto al training
        Da personalizzare in base al tuo caso specifico
        """
        # Esempio di implementazione - da adattare ai tuoi dati
        features = torch.FloatTensor(df.iloc[:, :-1].values)  # tutte le colonne tranne l'ultima
        targets = torch.FloatTensor(df.iloc[:, -1].values)    # ultima colonna
        return torch.utils.data.TensorDataset(features, targets)

    def save_checkpoint(self, filename):
        """Salva il checkpoint del modello"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename)

    def print_training_summary(self, history, total_time):
        """Stampa il riepilogo del training"""
        print("\n=== Training Summary ===")
        print(f"Total training time: {total_time:.2f} seconds")
        print("\nPer-dataset results:")
        for entry in history:
            print(f"\nDataset: {entry['dataset']}")
            print(f"Best loss: {entry['best_loss']:.4f}")
            print(f"Training time: {entry['time']:.2f} seconds")


if __name__ == "__main__":
    # Esempio di utilizzo
    model = YourModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # 2. Inizializza il data handler
    data_handler = HDF5Reader('/path/to/your/datasets')

    # 3. Crea il training manager
    trainer = TrainingManager(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        data_handler=data_handler,
        batch_size=32,
        num_epochs=10,
        early_stopping_patience=5
    )

    # 4. Avvia il training
    trainer.train_all_datasets()
















"""
class HDF5Loader(DataLoader):
    def __init__(self, database_path, batch_size, shuffle, num_workers):
        self.database_path = database_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.dataset = HDF5Reader(self.database_path)
        self.data = self.dataset.load_dataset()
        super(HDF5Loader, self).__init__(self.data, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data.iloc[idx] """