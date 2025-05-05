import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        # Fully connected layer to map LSTM output to original feature space
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_size]
        
        # LSTM output
        lstm_out, _ = self.lstm(x)
        
        # Convert LSTM output to predictions
        # This will predict the next value for each feature
        predictions = self.fc(lstm_out)
        
        return predictions
