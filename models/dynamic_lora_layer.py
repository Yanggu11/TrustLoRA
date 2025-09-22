import math

import torch
import torch.nn as nn


class DynamicLoRALayer(nn.Module):
    def __init__(self, hidden_size: int, r: int, hypernet: nn.Module, layer_id: int, use_fixed_A: bool = True, hypernet_use_batches: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.r = r
        self.hypernet = hypernet
        self.layer_id = layer_id
        self.use_fixed_A = use_fixed_A
        self.hypernet_use_batches = hypernet_use_batches

        self.weight = torch.tensor(0.0, dtype=self.hypernet.fc1.weight.dtype)

        self.A = torch.empty((self.hidden_size, self.r)) 
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.hypernet_use_batches:
            if self.use_fixed_A:
                A = self.A.to(x.device)
            else:
                A = torch.empty((self.hidden_size, self.r)).to(x.device)
                nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            B = self.hypernet(A, self.layer_id)  # A: [hidden, r], B: [r, hidden]
        else:
            A, B = self.hypernet.use_precomputed(self.layer_id) # A: [hidden, r], B: [r, hidden]

        output = torch.matmul(torch.matmul(x, A), B)  # returns [batch, seq_len, hidden]
        return output
