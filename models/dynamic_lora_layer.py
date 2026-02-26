import torch
import torch.nn as nn


class DynamicLoRALayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        r: int,
        hypernet: nn.Module,
        layer_id: int,
        hypernet_use_batches: bool = False,
        initial_A: torch.Tensor = None,
        initial_B: torch.Tensor = None,
        noise_type_A: str = "replace",
        noise_type_B: str = "replace",
        noise_alpha: float = 0.5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.r = r
        self.hypernet = hypernet
        self.layer_id = layer_id
        self.hypernet_use_batches = hypernet_use_batches

        self.register_buffer("initial_A", initial_A)
        self.register_buffer("initial_B", initial_B)

        self.noise_type_A = noise_type_A
        self.noise_type_B = noise_type_B

        self.alpha = noise_alpha

        self.weight = torch.tensor(0.0, dtype=next(self.hypernet.parameters()).dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.initial_A.device != x.device:
            self.initial_A = self.initial_A.to(x.device)
            self.initial_B = self.initial_B.to(x.device)

        if not self.hypernet_use_batches:
            A_hypernet, B_hypernet = self.hypernet(
                self.layer_id, device=x.device
            )  # A: [hidden, r], B: [r, hidden]
        else:
            A_hypernet, B_hypernet = self.hypernet.use_precomputed(
                self.layer_id
            )  # A: [hidden, r], B: [r, hidden]

        if self.noise_type_A == "replace":
            A = A_hypernet
        elif self.noise_type_A == "add":
            A = (1 - self.alpha) * A_hypernet + self.alpha * self.initial_A
        elif self.noise_type_A == "multiply":
            A = A_hypernet * self.initial_A

        if self.noise_type_B == "replace":
            B = B_hypernet
        elif self.noise_type_B == "add":
            B = (1 - self.alpha) * B_hypernet + self.alpha * self.initial_B
        elif self.noise_type_B == "multiply":
            B = B_hypernet * self.initial_B

        output = torch.matmul(torch.matmul(x, A), B)  # returns [batch, seq_len, hidden]
        return output
