import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

class Mine(nn.Module):
    def __init__(self, config, backbone):
        super().__init__()
        
        # Initialize the base DIT model
        self.backbone = backbone
        self.config = config
        
        # Get hidden dimension from config

        hidden_size = config.model.hidden_size

        # Define regression head options
        if config.model.regression_type == "attention":
            # Attention-based pooling
            self.regression_head = AttentionPooling(hidden_size)
        elif config.model.regression_type == "gap":
            self.regression_head = nn.Sequential(
                GlobalPooling(),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.SiLU(),
                nn.Linear(hidden_size // 2, 1)
            )
        elif config.model.regression_type == "mlp":
            self.regression_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.SiLU(),
                nn.Linear(hidden_size // 2, 1)
            )
        else:
            raise NotImplementedError(f"Regression {config.model.regression_type} not implemented")

    def forward(self, indices, sigma):
        # Get sequence representations from DIT model
        # We need to get hidden states before the final output layer
        if isinstance(self.backbone, transformers.PreTrainedModel):
            last_hidden_state = self.backbone(indices, sigma, output_hidden_states=True, return_dict=True).hidden_states[-1]
        else:
            last_hidden_state = self.backbone(indices, sigma)

        try:
            regression_output = self.regression_head(last_hidden_state)
        except:
            raise UserWarning(f"Incompatible shapes: {last_hidden_state.shape}, {self.config.model.hidden_size}")

        return regression_output

# Pooling options
class GlobalPooling(nn.Module):
    """Global average pooling over sequence dimension"""
    def forward(self, x):
        # x shape: [batch_size, seq_len, hidden_size]
        return torch.mean(x, dim=1)  # -> [batch_size, hidden_size]

class AttentionPooling(nn.Module):
    """Learn weights to attend to different positions in the sequence"""
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.regression = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, hidden_size]
        
        # Compute attention weights
        weights = self.attention(x)  # [batch_size, seq_len, 1]
        weights = F.softmax(weights, dim=1)
        
        # Apply attention weights
        context = torch.sum(weights * x, dim=1)  # [batch_size, hidden_size]
        
        # Final regression
        return self.regression(context)  # [batch_size, 1]