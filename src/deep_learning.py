"""
Deep Learning Models Module
============================
Sequence models for wellbore geology prediction:
  - LSTMModel     : Bidirectional LSTM with dropout
  - TransformerModel : Positional-encoding Transformer encoder

These models are designed to capture depth-sequential patterns
in LWD / drilling time-series data along horizontal wellbores.

Author: Md Ashraf
"""

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed. Deep learning models will not be available.")


# ---------------------------------------------------------------------------
# Guard decorator
# ---------------------------------------------------------------------------

def _require_torch(cls):
    """Class decorator that raises ImportError if torch is not installed."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for deep learning models. "
            "Install it with: pip install torch"
        )
    return cls


# ---------------------------------------------------------------------------
# LSTM Model
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class LSTMModel(nn.Module):
        """Bidirectional LSTM for depth-sequential wellbore prediction.

        Architecture:
            Input  →  (optional) Input projection
                    →  BiLSTM layers with dropout
                    →  Last hidden state
                    →  Fully-connected output head

        Args:
            input_size: Number of input features per depth step.
            hidden_size: LSTM hidden state dimension.
            num_layers: Number of stacked LSTM layers.
            output_size: Number of output targets (1 for regression).
            dropout: Dropout probability between LSTM layers.
            bidirectional: Whether to use bidirectional LSTM.
        """

        def __init__(
            self,
            input_size: int,
            hidden_size: int = 128,
            num_layers: int = 3,
            output_size: int = 1,
            dropout: float = 0.2,
            bidirectional: bool = True,
        ) -> None:
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.bidirectional = bidirectional
            self.num_directions = 2 if bidirectional else 1

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
                batch_first=True,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size * self.num_directions, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, output_size),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
            """Forward pass.

            Args:
                x: Input tensor of shape (batch_size, seq_len, input_size).

            Returns:
                Output tensor of shape (batch_size, output_size).
            """
            lstm_out, _ = self.lstm(x)        # (B, T, H * directions)
            last_out = lstm_out[:, -1, :]     # Take last timestep
            out = self.dropout(last_out)
            return self.fc(out)

    # --------------------------------------------------------------------------
    # Positional Encoding
    # --------------------------------------------------------------------------

    class PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding for Transformer input.

        Args:
            d_model: Embedding dimension.
            dropout: Dropout probability.
            max_len: Maximum sequence length supported.
        """

        def __init__(
            self,
            d_model: int,
            dropout: float = 0.1,
            max_len: int = 5000,
        ) -> None:
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)

            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
            )
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)             # (1, max_len, d_model)
            self.register_buffer("pe", pe)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
            """Add positional encoding to input embeddings.

            Args:
                x: Input tensor of shape (batch_size, seq_len, d_model).

            Returns:
                Positionally encoded tensor.
            """
            x = x + self.pe[:, : x.size(1), :]  # type: ignore[index]
            return self.dropout(x)

    # --------------------------------------------------------------------------
    # Transformer Model
    # --------------------------------------------------------------------------

    class TransformerModel(nn.Module):
        """Transformer encoder for wellbore depth-sequence modelling.

        Architecture:
            Input  →  Linear projection to d_model
                    →  Positional encoding
                    →  N × TransformerEncoderLayer
                    →  Global average pooling
                    →  Fully-connected output head

        Args:
            input_size: Number of input features per depth step.
            d_model: Transformer embedding dimension (must be divisible by nhead).
            nhead: Number of attention heads.
            num_encoder_layers: Number of Transformer encoder layers.
            dim_feedforward: Dimension of the FFN inside each encoder layer.
            dropout: Dropout probability.
            output_size: Number of output targets.
        """

        def __init__(
            self,
            input_size: int,
            d_model: int = 128,
            nhead: int = 8,
            num_encoder_layers: int = 4,
            dim_feedforward: int = 512,
            dropout: float = 0.1,
            output_size: int = 1,
        ) -> None:
            super().__init__()

            self.input_proj = nn.Linear(input_size, d_model)
            self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="relu",
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_encoder_layers,
            )

            self.fc = nn.Sequential(
                nn.Linear(d_model, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, output_size),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
            """Forward pass.

            Args:
                x: Input tensor of shape (batch_size, seq_len, input_size).

            Returns:
                Output tensor of shape (batch_size, output_size).
            """
            x = self.input_proj(x)           # (B, T, d_model)
            x = self.pos_encoder(x)
            x = self.transformer_encoder(x)  # (B, T, d_model)
            x = x.mean(dim=1)                # Global average pooling over T
            return self.fc(x)

    # --------------------------------------------------------------------------
    # Sequence Dataset
    # --------------------------------------------------------------------------

    class WellboreSequenceDataset(torch.utils.data.Dataset):
        """PyTorch Dataset for depth-windowed wellbore sequences.

        Each sample is a fixed-length window of depth rows extracted from
        a wellbore log. Stride controls the overlap between windows.

        Args:
            features: Feature array of shape (N, n_features).
            targets: Target array of shape (N,).
            window_size: Number of depth steps per sequence window.
            stride: Step size between consecutive windows.
        """

        def __init__(
            self,
            features: "torch.Tensor",  # type: ignore[name-defined]
            targets: "torch.Tensor",  # type: ignore[name-defined]
            window_size: int = 50,
            stride: int = 10,
        ) -> None:
            self.features = features
            self.targets = targets
            self.window_size = window_size
            self.stride = stride
            self.indices = list(range(0, len(features) - window_size + 1, stride))

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, idx: int):
            start = self.indices[idx]
            end = start + self.window_size
            x = self.features[start:end]                # (window_size, n_features)
            y = self.targets[end - 1]                   # Target at the final depth step of the window
            return x, y

else:
    # Stub classes when torch is not installed

    class LSTMModel:  # type: ignore[no-redef]
        """Stub: PyTorch not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required. Run: pip install torch")

    class TransformerModel:  # type: ignore[no-redef]
        """Stub: PyTorch not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required. Run: pip install torch")

    class WellboreSequenceDataset:  # type: ignore[no-redef]
        """Stub: PyTorch not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required. Run: pip install torch")
