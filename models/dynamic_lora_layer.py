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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A = torch.randn((self.r, self.hidden_size)).to(x.device)

        B = self.hypernet(A, self.layer_id)  # A: [r, hidden], B: [hidden, r]
        output = torch.matmul(torch.matmul(x, B), A)  # returns [batch, seq_len, hidden]
        return output