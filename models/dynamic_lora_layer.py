import torch
import torch.nn as nn


class DynamicLoRALayer(nn.Module):
    def __init__(self, hidden_size, r, hypernet: nn.Module):
        super().__init__()
        self.hidden_size = hidden_size
        self.r = r
        self.hypernet = hypernet

        self.weight = torch.tensor(0.0, dtype=self.hypernet.fc1.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            A = torch.randn((self.r, self.hidden_size)).to(x.device)

            B = self.hypernet(A)  # A: [r, hidden], B: [hidden, r]
            output = torch.matmul(torch.matmul(x, B), A)  # returns [batch, seq_len, hidden]
        else:
            with torch.no_grad():
                A = torch.randn((self.r, self.hidden_size)).to(x.device)

                B = self.hypernet(A)  # A: [r, hidden], B: [hidden, r]
                output = torch.matmul(torch.matmul(x, B), A)  # returns [batch, seq_len, hidden]
        return output