import math

import torch
import torch.nn as nn


class DynamicLoRALayer(nn.Module):
    def __init__(self, hidden_size: int, r: int, hypernet: nn.Module, layer_id: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.r = r
        self.hypernet = hypernet
        self.layer_id = layer_id

        self.weight = torch.tensor(0.0, dtype=self.hypernet.fc1.weight.dtype)

        self.A = torch.empty((self.hidden_size, self.r))  # uninitialized tensor
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A = self.A.to(x.device)
        B = self.hypernet(A, self.layer_id)  # A: [hidden, r], B: [r, hidden]

        output = torch.matmul(torch.matmul(x, A), B)  # returns [batch, seq_len, hidden]
        return output
