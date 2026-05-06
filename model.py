import torch
import torch.nn as nn
import numpy as np

class PedestrianPredictor(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, num_layers=2, dropout=0.1):
        """
        GRU-based encoder-decoder model for pedestrian intent and trajectory prediction.
        
        Args:
            input_size (int): Number of features per timestep (default: 10)
            hidden_size (int): Hidden size of GRU layers
            num_layers (int): Number of GRU layers
            dropout (float): Dropout rate for regularization
        """
        super(PedestrianPredictor, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU Encoder (2-layer)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output heads
        # Intent head: Linear -> Sigmoid for probability output
        self.intent_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Trajectory head: Linear layer outputting 8 coordinates (4 future bbox centers)
        self.trajectory_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 8)  # 4 boxes × 2 coords (cx, cy)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            tuple: (intent_output, trajectory_output)
                - intent_output: (batch_size, 1) - crossing probability
                - trajectory_output: (batch_size, 8) - future bbox centers
        """
        # Encode sequence with GRU
        # gru_output: (batch_size, sequence_length, hidden_size)
        # hidden: (num_layers, batch_size, hidden_size)
        gru_output, hidden = self.gru(x)
        
        # Use the last hidden state as context vector
        # Take the last layer's hidden state
        context_vector = hidden[-1]  # (batch_size, hidden_size)
        
        # Pass through output heads
        intent_output = self.intent_head(context_vector)      # (batch_size, 1)
        trajectory_output = self.trajectory_head(context_vector)  # (batch_size, 8)
        
        return intent_output, trajectory_output
    
    def predict_single(self, x):
        """
        Prediction method for single sample (used in inference).
        
        Args:
            x (torch.Tensor): Input tensor of shape (1, sequence_length, input_size)
            
        Returns:
            tuple: (intent_prob, trajectory_coords)
        """
        self.eval()
        with torch.no_grad():
            intent_output, trajectory_output = self.forward(x)
            
            # Convert to numpy and squeeze batch dimension
            intent_prob = intent_output.squeeze().item()
            trajectory_coords = trajectory_output.squeeze().numpy()
            
            return intent_prob, trajectory_coords

def save_model(model, filepath):
    """Save model state dict to file."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': model.input_size,
            'hidden_size': model.hidden_size,
            'num_layers': model.num_layers
        }
    }, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath, device='cpu'):
    """Load model from file."""
    checkpoint = torch.load(filepath, map_location=device)
    
    # Extract model configuration
    config = checkpoint['model_config']
    
    # Create model instance
    model = PedestrianPredictor(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers']
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Model loaded from {filepath}")
    return model

# Test the model
if __name__ == "__main__":
    # Test model instantiation and forward pass
    model = PedestrianPredictor()
    
    # Create dummy input: batch_size=2, sequence_length=16, input_size=10
    dummy_input = torch.randn(2, 16, 10)
    
    intent_output, trajectory_output = model(dummy_input)
    
    print("Model test successful!")
    print(f"Intent output shape: {intent_output.shape}")      # Should be (2, 1)
    print(f"Trajectory output shape: {trajectory_output.shape}") # Should be (2, 8)
    print(f"Intent values: {intent_output.squeeze().tolist()}")
    print(f"Trajectory values: {trajectory_output[0].tolist()}")