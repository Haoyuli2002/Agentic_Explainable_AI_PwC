import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class LoanDefaultModel(nn.Module):
    def __init__(self, input_size=33, hidden_size=32, num_layers=1, dropout=0.1):
        super(LoanDefaultModel, self).__init__()
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Batch Normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size*2)

        # Dense Layers
        self.fc1 = nn.Linear(hidden_size*2, 16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        # x: (batch, years, features)
        # lstm_out: (batch, years, hidden_size * num_directions) = (batch, 13, 32*2)
        # Hidden state for each time step
        # h_n: final hidden state for each direction (num_layers * num_directions, batch, hidden_size) = (2, batch, 64)
        # c_n: cell, long-term memory unit.
        lstm_out, (hn, cn) = self.lstm(x)

        hn_forward = hn[-2, :, :]  # (batch, hidden_size)
        hn_backward = hn[-1, :, :] # (batch, hidden_size)
        last_out = torch.cat([hn_forward, hn_backward], dim=1) # (batch, hidden_size*2)
        last_out = self.batch_norm(last_out)

        output = self.fc2(self.dropout(self.relu(self.fc1(last_out))))
        return output

class LoanDefaultDataset(Dataset):
    """
    Dataset for final year prediction
    
    - sequences: input data with shape (n_loans, max_seq_length, num_features)
    - targets: output data with shape (n_loans, 1)
    """

    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        return self.sequences[index], self.targets[index]
