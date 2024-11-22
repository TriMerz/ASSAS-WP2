#!/usr/bin/env python3

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPBaseline, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = torch.relu(out)
        out = self.fc2(out)
        return out



class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = torch.relu(out)
        out = self.fc2(out)
        out = torch.relu(out)
        out = self.fc3(out)
        return out



class DeeperMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, sequence_length, forecast_steps=1):
        super(DeeperMLP, self).__init__()
        
        self.flatten = nn.Flatten()
        flattened_input = input_size * sequence_length
        
        self.network = nn.Sequential(
            nn.Linear(flattened_input, hidden_size * 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size * 4),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size * 2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 4),
            
            nn.Linear(hidden_size // 4, output_size * forecast_steps)
        )
        self.forecast_steps = forecast_steps

    def forward(self, x):
        x = self.flatten(x)
        out = self.network(x)
        return out.view(out.size(0), self.forecast_steps, -1)



class DeepMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, sequence_length):
        super(DeepMLP, self).__init__()
        
        self.flatten = nn.Flatten()
        flattened_input = input_size * sequence_length
        
        self.network = nn.Sequential(
            nn.Linear(flattened_input, hidden_size * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size * 2),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)