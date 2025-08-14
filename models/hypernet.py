import torch
import torch.nn as nn


class LoRAHyperNet(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, lora_r, num_of_embeddings=2, embedding_dim=8, embedding_input_only=False, large_model=False
    ):
        super().__init__()

        self.lora_r = lora_r
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_of_embeddings = num_of_embeddings
        self.embedding_input_only = embedding_input_only
        self.large_model = large_model

        if self.embedding_input_only:
            self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        else:
            self.fc1 = nn.Linear(lora_r * input_dim + embedding_dim, hidden_dim)

        if self.large_model: 
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, hidden_dim)
            self.fc4 = nn.Linear(hidden_dim, input_dim * lora_r)
        else:
            self.fc2 = nn.Linear(hidden_dim, input_dim * lora_r)

        self.embedding = nn.Embedding(num_of_embeddings, embedding_dim)

    def forward(self, inpt, layer_id):
        layer_id = torch.tensor(layer_id).to(inpt.device)

        layer_embedding = self.embedding(layer_id)
        if self.embedding_input_only:
            flat = layer_embedding
        else:
            flat = torch.cat((layer_embedding, inpt.flatten()), dim=0)

        h = torch.relu(self.fc1(flat))
        if self.large_model:
            h = torch.relu(self.fc2(h))
            h = torch.relu(self.fc3(h))
            opt = self.fc4(h).view(
                self.lora_r, self.input_dim
            )
        else:
            opt = self.fc2(h).view(
                self.lora_r, self.input_dim
            )

        return opt


# class LoRAHyperNetEmbeddingInput(nn.Module):
#     def __init__(
#         self, input_dim, hidden_dim, lora_r, num_of_embeddings=2, embedding_dim=8
#     ):
#         super().__init__()

#         self.lora_r = lora_r
#         self.hidden_dim = hidden_dim
#         self.input_dim = input_dim
#         self.num_of_embeddings = num_of_embeddings

#         self.fc1 = nn.Linear(embedding_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, input_dim * lora_r)

#         self.embedding = nn.Embedding(num_of_embeddings, embedding_dim)

#     def forward(self, inpt, layer_id):

#         layer_id = torch.tensor(layer_id).to(next(iter(self.parameters())).device)

#         layer_embedding = self.embedding(layer_id)

#         h = torch.relu(self.fc1(layer_embedding))
#         opt = self.fc2(h).view(
#             self.lora_r, self.input_dim
#         )  # B shape: [r, in_dim] assuming that target output dimension is the same as input dimension

#         return opt
