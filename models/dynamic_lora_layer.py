import math

import torch
import torch.nn as nn


class DynamicLoRALayer(nn.Module):
    def __init__(self, hidden_size: int, r: int, hypernet: nn.Module, layer_id: int, hypernet_use_batches: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.r = r
        self.hypernet = hypernet
        self.layer_id = layer_id
        self.hypernet_use_batches = hypernet_use_batches

        self.weight = torch.tensor(0.0, dtype=self.hypernet.fc1.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if not self.hypernet_use_batches:
            A, B = self.hypernet(self.layer_id, device=x.device)  # A: [hidden, r], B: [r, hidden]
        else:
            A, B = self.hypernet.use_precomputed(self.layer_id) # A: [hidden, r], B: [r, hidden]

        output = torch.matmul(torch.matmul(x, A), B)  # returns [batch, seq_len, hidden]
        return output
