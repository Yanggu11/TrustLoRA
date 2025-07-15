import torch
import torch.nn as nn


class LoRAHyperNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, lora_r):
        super().__init__()

        self.lora_r = lora_r
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.fc1 = nn.Linear(lora_r * input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim * lora_r)

    def forward(self, inpt):
        flat = inpt.view(-1)
        h = torch.relu(self.fc1(flat))
        opt = self.fc2(h).view(inpt.shape[1], inpt.shape[0])  # B shape: [in_dim, r]

        return opt