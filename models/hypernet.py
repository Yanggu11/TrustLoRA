import torch
import torch.nn as nn
import math


class LoRAHyperNet(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, lora_r, num_of_embeddings=2, embedding_dim=8,
        use_fixed_A=False, use_batches=True, embedding_input_only=False, large_model=False
    ):
        super().__init__()

        self.lora_r = lora_r
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_of_embeddings = num_of_embeddings
        self.use_fixed_A = use_fixed_A
        self.use_batches = use_batches
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

        if self.use_fixed_A or self.use_batches:
            self.A_matrices = [torch.empty((self.input_dim, self.lora_r)) for _ in range(num_of_embeddings)]
            for A_matrix in self.A_matrices:
                nn.init.kaiming_uniform_(A_matrix, a=math.sqrt(5))
        else:
            self.A_matrices = None

        if self.use_batches:
            self.B_matrices = [torch.empty((self.lora_r, self.input_dim)) for _ in range(num_of_embeddings)]
        else:
            self.B_matrices = None

    def forward(self, layer_id, device="cpu"):
        layer_id = torch.tensor(layer_id).to(device)

        layer_embedding = self.embedding(layer_id)

        if self.use_fixed_A:
            A = self.A_matrices[layer_id].to(device)
        else:
            A = torch.empty((self.input_dim, self.lora_r)).to(device)
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))

        if self.embedding_input_only:
            flat = layer_embedding
        else:
            flat = torch.cat((layer_embedding, A.flatten()), dim=0)

        h = torch.relu(self.fc1(flat))
        if self.large_model:
            h = torch.relu(self.fc2(h))
            h = torch.relu(self.fc3(h))
            B = self.fc4(h).view(self.lora_r, self.input_dim)
        else:
            B = self.fc2(h).view(self.lora_r, self.input_dim)

        return A, B

    def precompute(self, device="cpu"):
        print("PRECOMPUTINGNGGGG ")
        if not self.use_batches:
            raise RuntimeError("precompute() called but use_batches is disabled.")

        layer_ids = torch.arange(self.num_of_embeddings, device=device)
        embeddings = self.embedding(layer_ids)  # (N, embedding_dim)

        if self.use_fixed_A:
            As = torch.stack([A.to(device) for A in self.A_matrices], dim=0)  # (N, in_dim, r)
        else:
            As = torch.empty((self.num_of_embeddings, self.input_dim, self.lora_r), device=device)
            for i in range(self.num_of_embeddings):
                nn.init.kaiming_uniform_(As[i], a=math.sqrt(5))
            self.A_matrices = [As[i] for i in range(self.num_of_embeddings)]

        if self.embedding_input_only:
            flat = embeddings  # (N, embed_dim)
        else:
            As_flat = As.view(self.num_of_embeddings, -1)  # (N, in_dim*r)
            flat = torch.cat([embeddings, As_flat], dim=1)  # (N, embed_dim + in_dim*r)

        h = torch.relu(self.fc1(flat))
        if self.large_model:
            h = torch.relu(self.fc2(h))
            h = torch.relu(self.fc3(h))
            Bs = self.fc4(h).view(self.num_of_embeddings, self.lora_r, self.input_dim)
        else:
            Bs = self.fc2(h).view(self.num_of_embeddings, self.lora_r, self.input_dim)

        self.A_matrices = [As[i].detach().clone() for i in range(self.num_of_embeddings)]
        self.B_matrices = [Bs[i].detach().clone() for i in range(self.num_of_embeddings)]

    def use_precomputed(self, layer_id):
        print(f"USING PRECOMPUTED {layer_id}")

        if self.A_matrices is None or self.B_matrices is None:
            raise RuntimeError("No precomputed matrices found. Call precompute() first.")

        return self.A_matrices[layer_id], self.B_matrices[layer_id]
