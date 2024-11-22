#!/usr/bin/env python3
 
"""
File contains the training function for one epoch.
The function is called by the train method of the TrainingManager class.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
import time
from tqdm import tqdm



def train_step(model,
               train_dataloader, # comprende sia i dati che i target (?)
               optimizer,
               device = 'cuda' if torch.cuda.is_available() else 'cpu',
               # memory = False,
               memory = False,
               verbose = True,
               lr_scheduler = None):
    """
    Train model for one epoch.
    """

    model.train()
    total_loss = 0
    num_batches = 0
    
    # Progress bar per i batch
    pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        output = model(data
                       # altri dati utili che servono per il mio modello...
                       )
        
        """
        # Metodo per mantenere memoria di alcuni dati
        if memory:
            saved_items = memory['saved_items']
        """

        # Calcolo della loss
        loss = loss_function(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        # Aggiornamento della loss
        batch_loss += loss.item()
        
        # Aggiornamento della progress bar
        pbar.set_postfix({'loss': total_loss/num_batches})
    
    epoch_loss = total_loss/num_batches

    if verbose:
        print(f'Epoch {epoch+1} - Loss: {epoch_loss}')
        
    return epoch_loss