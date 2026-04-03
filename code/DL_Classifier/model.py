"""
model.py
========================================================
LSTM model definition for sleep stage classification
Contains base LSTM, Bidirectional LSTM, and LSTM with Attention mechanism
Supports masking mechanism
"""

import torch
import torch.nn as nn
from typing import Optional


class SleepStageLSTM(nn.Module):
    """LSTM model for sleep stage classification (supports masking)"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.5,
        use_layer_norm: bool = True,
        use_mask: bool = True  # Added: whether to use masking
    ):
        """
        Args:
            input_size: Input feature dimension
            hidden_size: LSTM hidden layer size
            num_layers: Number of LSTM layers
            num_classes: Number of classification classes (3 for multi-class, 2 for binary)
            dropout: Dropout probability
            use_layer_norm: Whether to use layer normalization
            use_mask: Whether to use masking mechanism
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.use_mask = use_mask
        
        # Input projection layer (optional)
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Calculate LSTM output size
        lstm_output_size = hidden_size
        
        # Classification head
        classifier_input_size = lstm_output_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_size) if use_layer_norm else nn.Identity(),
            nn.Dropout(dropout),
            nn.Linear(classifier_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def _apply_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply mask to input sequence"""
        if mask is not None and self.use_mask:
            # Zero out features of invalid time steps
            mask_expanded = mask.unsqueeze(-1).float()  # (batch_size, seq_len, 1)
            x = x * mask_expanded
        return x
    
    def _get_last_valid_output(self, lstm_out: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Get LSTM output of the last valid time step"""
        if mask is not None and self.use_mask:
            # Get the index of the last valid time step for each sequence
            lengths = mask.sum(dim=1).long() - 1  # Index of the last valid position
            batch_size = lstm_out.size(0)
            last_output = lstm_out[torch.arange(batch_size), lengths]
        else:
            # Traditional method: use the last time step
            last_output = lstm_out[:, -1, :]
        return last_output
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: Input sequence (batch, seq_len, input_size)
            mask: Mask (batch, seq_len) - True means valid, False means invalid
            return_attention: Whether to return attention weights
        
        Returns:
            If return_attention=False: Prediction results (batch, num_classes)
            If return_attention=True: (Prediction results, attention weights)
        """
        batch_size = x.size(0)
        
        # Apply mask
        x = self._apply_mask(x, mask)
        
        # Input projection
        x = self.input_projection(x)  # (batch, seq_len, hidden_size)
        
        # LSTM forward propagation
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, lstm_output_size)
        

        # Use the output of the last valid time step
        output = self._get_last_valid_output(lstm_out, mask)
        attention_weights = None
        
        # Classification
        logits = self.classifier(output)  # (batch, num_classes)
        
        if return_attention and attention_weights is not None:
            return logits, attention_weights
        return logits


class SleepStageGRU(nn.Module):
    """GRU model for sleep stage classification (supports masking)"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        num_classes: int = 3,
        dropout: float = 0.5,
        use_mask: bool = True  # Added: whether to use masking
    ):
        super().__init__()
        
        self.use_mask = use_mask
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        gru_output_size = hidden_size
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def _apply_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply mask to input sequence"""
        if mask is not None and self.use_mask:
            mask_expanded = mask.unsqueeze(-1).float()
            x = x * mask_expanded
        return x
    
    def _get_last_valid_output(self, gru_out: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Get GRU output of the last valid time step"""
        if mask is not None and self.use_mask:
            lengths = mask.sum(dim=1).long() - 1
            batch_size = gru_out.size(0)
            last_output = gru_out[torch.arange(batch_size), lengths]
        else:
            last_output = gru_out[:, -1, :]
        return last_output
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Apply mask
        x = self._apply_mask(x, mask)
        
        gru_out, _ = self.gru(x)
        output = self._get_last_valid_output(gru_out, mask)
        return self.classifier(output)


class SleepStageTransformer(nn.Module):
    """Transformer-based model for sleep stage classification (supports masking)"""
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 32,
        nhead: int = 4, 
        num_layers: int = 1,
        num_classes: int = 2,
        dropout: float = 0.25,
        max_seq_length: int = 60,
        use_mask: bool = True  # Added: whether to use masking
    ):
        super().__init__()

        self.use_mask = use_mask
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, max_seq_length, d_model)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
    
    def _apply_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply mask to input sequence"""
        if mask is not None and self.use_mask:
            # Add a True value for the CLS token
            cls_mask = torch.ones(x.size(0), 1, dtype=torch.bool, device=x.device)
            extended_mask = torch.cat([cls_mask, mask], dim=1)
            
            # Create mask format required by Transformer
            src_key_padding_mask = ~extended_mask  # Transformer requires True for positions to be masked
            return x, src_key_padding_mask
        return x, None
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Apply mask
        x, src_key_padding_mask = self._apply_mask(x, mask)
        
        # Input projection
        x = self.input_projection(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :seq_len+1]
        
        # Transformer encoding (pass in mask)
        if src_key_padding_mask is not None:
            x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        else:
            x = self.transformer(x)
        
        # Use CLS token output for classification
        cls_output = x[:, 0]
        
        return self.classifier(cls_output)


class SleepStageMLP(nn.Module):
    """Multi-Layer Perceptron (MLP) baseline model - used to compare with temporal models"""

    def __init__(
            self,
            input_size: int,
            hidden_sizes: list = [64, 16, 4],
            num_classes: int = 3,
            dropout: float = 0.1,
            use_batch_norm: bool = True,
            activation: str = 'gelu',
            use_mask: bool = True  # Added: whether to use masking
    ):
        """
        Args:
            input_size: Feature dimension for each time step
            hidden_sizes: List of hidden layer sizes
            num_classes: Number of classification classes
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
            activation: Activation function type ('relu', 'gelu', 'leaky_relu')
            use_mask: Whether to use masking mechanism
        """
        super().__init__()

        self.use_mask = use_mask
        self.input_size = input_size
        self.flattened_size = input_size

        # Select activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build network layers
        layers = []
        current_size = self.flattened_size

        for hidden_size in hidden_sizes:
            # Linear layer
            layers.append(nn.Linear(current_size, hidden_size))

            # Batch normalization (optional)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))

            # Activation function
            layers.append(self.activation)

            # Dropout
            layers.append(nn.Dropout(dropout))

            current_size = hidden_size

        # Output layer
        layers.append(nn.Linear(current_size, num_classes))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _apply_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply mask to input sequence (For MLP, we only use features from valid windows)"""
        if mask is not None and self.use_mask:
            # For MLP, we only use the last valid window
            batch_size = x.size(0)
            valid_outputs = []
            for i in range(batch_size):
                # Find the index of the last valid window
                valid_indices = mask[i].nonzero(as_tuple=True)[0]
                if len(valid_indices) > 0:
                    last_valid_idx = valid_indices[-1]
                    valid_outputs.append(x[i, last_valid_idx])
                else:
                    # If no valid window, use zero vector
                    valid_outputs.append(torch.zeros_like(x[i, 0]))
            x = torch.stack(valid_outputs, dim=0)
            return x.unsqueeze(1)  # Maintain (batch, 1, input_size) dimension
        else:
            # If no mask, use the last time step
            return x[:, -1:]

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input sequence (batch, seq_len, input_size)
            mask: Mask (batch, seq_len) - True means valid, False means invalid

        Returns:
            Prediction results (batch, num_classes)
        """
        # Apply mask: only use valid window
        x = self._apply_mask(x, mask)
        
        batch_size = x.size(0)
        # Flatten input
        x_flat = x.view(batch_size, -1)  # (batch, input_size)

        # Pass through MLP network
        output = self.network(x_flat)

        return output

def create_model(
    model_type: str,
    input_size: int,
    num_classes: int,
    use_mask: bool = True,  # Added: whether to use masking
    **kwargs
) -> nn.Module:
    """
    Factory function to create models
    
    Args:
        model_type: "lstm", "gru", "transformer" or "mlp"
        input_size: Input feature dimension
        num_classes: Number of classification classes
        use_mask: Whether to use masking mechanism
        **kwargs: Other model parameters
    
    Returns:
        Model instance
    """
    if model_type == "lstm":
        return SleepStageLSTM(
            input_size=input_size,
            num_classes=num_classes,
            use_mask=use_mask,
            use_layer_norm=kwargs.get("use_layer_norm", True),
            dropout=kwargs.get("dropout", 0.25),
            num_layers=kwargs.get("num_layers", 1),
            hidden_size=kwargs.get("hidden_size", 32),
        )
    elif model_type == "gru":
        return SleepStageGRU(
            input_size=input_size,
            num_classes=num_classes,
            use_mask=use_mask,
            dropout=kwargs.get("dropout", 0.25),
            num_layers=kwargs.get("num_layers", 1),
            hidden_size=kwargs.get("hidden_size", 32),
            
        )
    elif model_type == "transformer":
        return SleepStageTransformer(
            input_size=input_size,
            num_classes=num_classes,
            use_mask=use_mask,
            dropout=kwargs.get("dropout", 0.25),
            max_seq_length=kwargs.get("max_seq_length", 60),
            nhead=kwargs.get("nhead", 4),
            num_layers=kwargs.get("num_layers", 1),
        )
    elif model_type == "mlp":
        return SleepStageMLP(
            input_size=input_size,
            num_classes=num_classes,
            use_mask=use_mask,
            dropout=kwargs.get("dropout", 0.1),
            hidden_sizes=kwargs.get("hidden_sizes", [64, 16, 4]),
            use_batch_norm=kwargs.get("use_batch_norm", True),
            activation=kwargs.get("activation", "gelu"),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")