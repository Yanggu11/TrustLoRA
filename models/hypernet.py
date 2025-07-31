import torch
import torch.nn as nn


class LoRAHyperNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, lora_r, num_of_embeddings=2, embedding_dim=8):
        super().__init__()

        self.lora_r = lora_r
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_of_embeddings = num_of_embeddings

        self.fc1 = nn.Linear(lora_r * input_dim + embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim * lora_r)

        self.embedding = nn.Embedding(num_of_embeddings, embedding_dim)

    def forward(self, inpt, layer_id):
        input_shape = inpt.shape
        layer_id = torch.tensor(layer_id).to(inpt.device)

        layer_embedding = self.embedding(layer_id)
        flat = torch.cat((layer_embedding, inpt.flatten()), dim=0)

        h = torch.relu(self.fc1(flat))
        opt = self.fc2(h).view(input_shape[1], input_shape[0])  # B shape: [r, in_dim]

        return opt