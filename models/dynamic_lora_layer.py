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

        self.A = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.A is None:
            self.A = torch.randn((self.hidden_size, self.r)).to(x.device)

        A = self.A

        B = self.hypernet(A, self.layer_id)  # A: [hidden, r], B: [r, hidden]
        output = torch.matmul(torch.matmul(x, A), B)  # returns [batch, seq_len, hidden]
        return output