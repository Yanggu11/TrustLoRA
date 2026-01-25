import math

import torch
import torch.nn as nn

from utils.one_hot_encoding import OneHotEncoder


class LoRAHyperNet(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        lora_r,
        use_embedding=True,
        num_of_embeddings=2,
        embedding_dim=8,
        hypernet_A_matrix="random",
        use_batches=True,
        embedding_input_only=False,
        large_model=False,
    ):
        super().__init__()

        self.lora_r = lora_r
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_of_embeddings = num_of_embeddings
        self.hypernet_A_matrix = hypernet_A_matrix
        self.use_batches = use_batches
        self.embedding_input_only = embedding_input_only
        self.large_model = large_model
        self.use_embedding = use_embedding

        self.activation = nn.GELU()

        assert not (not self.embedding_input_only and self.hypernet_A_matrix == "generated"), (
            "Cannot use A as input with generated A matrix."
        )

        if self.embedding_input_only:
            self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        else:
            self.fc1 = nn.Linear(lora_r * input_dim + embedding_dim, hidden_dim)

        output_dim = input_dim * lora_r if self.hypernet_A_matrix != "generated" else 2 * input_dim * lora_r

        if self.large_model:
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, hidden_dim)
            self.fc4 = nn.Linear(hidden_dim, output_dim)
        else:
            self.fc2 = nn.Linear(hidden_dim, output_dim)

        if self.use_embedding:
            self.embedding = nn.Embedding(num_of_embeddings, embedding_dim)
        else:
            self.one_hot = OneHotEncoder(num_of_embeddings)

        if self.hypernet_A_matrix == "fixed" or self.use_batches:
            self.A_matrices = [
                torch.empty((self.input_dim, self.lora_r))
                for _ in range(num_of_embeddings)
            ]
            for A_matrix in self.A_matrices:
                nn.init.kaiming_uniform_(A_matrix, a=math.sqrt(5))
        else:
            self.A_matrices = None

        if self.use_batches:
            self.B_matrices = [
                torch.empty((self.lora_r, self.input_dim))
                for _ in range(num_of_embeddings)
            ]
        else:
            self.B_matrices = None

    def forward(self, layer_id, device="cpu"):
        layer_id = torch.tensor(layer_id).to(device)

        if self.use_embedding:
            layer_embedding = self.embedding(layer_id)
        else:
            one_hot = self.one_hot.encode([layer_id])
            layer_embedding = torch.tensor(
                one_hot, dtype=torch.float32, device=device
            ).squeeze(0)

        if self.hypernet_A_matrix == "fixed":
            A = self.A_matrices[layer_id].to(device)
        elif self.hypernet_A_matrix == "random":
            A = torch.empty((self.input_dim, self.lora_r)).to(device)
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))

        if self.embedding_input_only:
            flat = layer_embedding
        else:
            flat = torch.cat((layer_embedding, A.flatten()), dim=0)

        h = self.activation(self.fc1(flat))
        if self.large_model:
            h = self.activation(self.fc2(h))
            h = self.activation(self.fc3(h))
            out = self.fc4(h)
        else:
            out = self.fc2(h)

        if self.hypernet_A_matrix == "generated":
            B = out[:self.lora_r * self.input_dim].view(self.lora_r, self.input_dim)
            A = out[self.lora_r * self.input_dim:].view(self.input_dim, self.lora_r)
        else:
            B = out.view(self.lora_r, self.input_dim)

        return A, B

    def precompute(self, device="cpu"):
        if not self.use_batches:
            raise RuntimeError("precompute() called but use_batches is disabled.")

        layer_ids = torch.arange(self.num_of_embeddings, device=device)
        embeddings = self.embedding(layer_ids)  # (N, embedding_dim)

        if self.hypernet_A_matrix == "fixed":
            As = torch.stack(
                [A.to(device) for A in self.A_matrices], dim=0
            )  # (N, in_dim, r)
        elif self.hypernet_A_matrix == "random":
            As = torch.empty(
                (self.num_of_embeddings, self.input_dim, self.lora_r), device=device
            )
            for i in range(self.num_of_embeddings):
                nn.init.kaiming_uniform_(As[i], a=math.sqrt(5))
            self.A_matrices = [As[i] for i in range(self.num_of_embeddings)]

        if self.embedding_input_only:
            flat = embeddings  # (N, embed_dim)
        else:
            As_flat = As.view(self.num_of_embeddings, -1)  # (N, in_dim*r)
            flat = torch.cat([embeddings, As_flat], dim=1)  # (N, embed_dim + in_dim*r)

        h = self.activation(self.fc1(flat))
        if self.large_model:
            h = self.activation(self.fc2(h))
            h = self.activation(self.fc3(h))
            out = self.fc4(h)
        else:
            out = self.fc2(h)

        if self.hypernet_A_matrix == "generated":
            Bs = out[:, : self.lora_r * self.input_dim].view(
                self.num_of_embeddings, self.lora_r, self.input_dim
            )
            As = out[:, self.lora_r * self.input_dim :].view(
                self.num_of_embeddings, self.input_dim, self.lora_r
            )
        else:
            Bs = out.view(self.num_of_embeddings, self.lora_r, self.input_dim)
            
        self.A_matrices = [
            As[i].detach().clone() for i in range(self.num_of_embeddings)
        ]
        self.B_matrices = [
            Bs[i] for i in range(self.num_of_embeddings)
        ]

    def use_precomputed(self, layer_id):

        if self.A_matrices is None or self.B_matrices is None:
            raise RuntimeError(
                "No precomputed matrices found. Call precompute() first."
            )

        return self.A_matrices[layer_id], self.B_matrices[layer_id]


class LoRAHyperNetTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        lora_r,
        use_embedding=True,
        num_of_embeddings=2,
        embedding_dim=8,
        hypernet_A_matrix="random",
        use_batches=True,
        embedding_input_only=False,
        nhead=4,
        num_layers=2,
    ):
        super().__init__()

        self.lora_r = lora_r
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.hypernet_A_matrix = hypernet_A_matrix
        self.num_of_embeddings = num_of_embeddings if self.hypernet_A_matrix == "generated" else num_of_embeddings * 2
        self.use_batches = use_batches
        self.embedding_input_only = embedding_input_only
        self.use_embedding = use_embedding

        self.activation = nn.GELU()

        assert not (not self.embedding_input_only and self.hypernet_A_matrix == "generated"), (
            "Cannot use A as input with generated A matrix."
        )

        if self.use_embedding:
            self.embedding = nn.Embedding(num_of_embeddings, embedding_dim)
        else:
            self.one_hot = OneHotEncoder(num_of_embeddings)

        if embedding_input_only:
            token_dim = embedding_dim
            output_dim = input_dim if not use_batches else input_dim * lora_r
        elif not use_batches:
            token_dim = embedding_dim + input_dim  # (each token = embedding + A[:,i])
            output_dim = input_dim
        else:
            token_dim = embedding_dim + input_dim * lora_r
            output_dim = input_dim * lora_r

        self.token_proj = nn.Linear(token_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.out_proj = nn.Linear(hidden_dim, output_dim)

        if self.hypernet_A_matrix == "fixed" or self.use_batches:
            self.A_matrices = [
                torch.empty((self.input_dim, self.lora_r))
                for _ in range(num_of_embeddings)
            ]
            for A_matrix in self.A_matrices:
                nn.init.kaiming_uniform_(A_matrix, a=math.sqrt(5))
        else:
            self.A_matrices = None

        if self.use_batches:
            self.B_matrices = [
                torch.empty((self.lora_r, self.input_dim))
                for _ in range(num_of_embeddings)
            ]
        else:
            self.B_matrices = None

    def forward(self, layer_id, device="cpu"):

        raise NotImplementedError("Use precompute() and use_precomputed() methods instead.")

        # layer_id = torch.tensor(layer_id).to(device)

        # if self.use_embedding:
        #     # ! This need to be fixed, bc we also want to have separate embeddings for each rank
        #     layer_embedding = self.embedding(layer_id)  # (embed_dim,)
        # else:
        #     one_hot = self.one_hot.encode([layer_id])
        #     layer_embedding = torch.tensor(
        #         one_hot, dtype=torch.float32, device=device
        #     ).squeeze(0)

        # if self.hypernet_A_matrix == "fixed":
        #     A = self.A_matrices[layer_id].to(device)  # (in_dim, r)
        # else:
        #     A = torch.empty((self.input_dim, self.lora_r), device=device)
        #     nn.init.kaiming_uniform_(A, a=math.sqrt(5))

        # tokens = []
        # for i in range(self.lora_r):
        #     if self.embedding_input_only:
        #         token = layer_embedding
        #     else:
        #         token = torch.cat([layer_embedding, A[:, i]], dim=0)
        #     tokens.append(token)

        # tokens = torch.stack(tokens, dim=0).unsqueeze(0)  # (1, r, token_dim)
        # tokens = self.token_proj(tokens)  # (1, r, hidden)

        # h = self.transformer(tokens)  # (1, r, hidden)

        # B = self.out_proj(h).squeeze(0)

        # return A, B

    def precompute(self, device="cpu"):
        if not self.use_batches:
            raise RuntimeError("precompute() called but use_batches is disabled.")

        layer_ids = torch.arange(self.num_of_embeddings, device=device)
        embeddings = self.embedding(layer_ids)  # (N, embed_dim)

        if self.hypernet_A_matrix == "fixed":
            As = torch.stack(
                [A.to(device) for A in self.A_matrices], dim=0
            )  # (N, in_dim, r)
        elif self.hypernet_A_matrix == "random":
            As = torch.empty(
                (self.num_of_embeddings, self.input_dim, self.lora_r), device=device
            )
            for i in range(self.num_of_embeddings):
                nn.init.kaiming_uniform_(As[i], a=math.sqrt(5))
            self.A_matrices = [As[i] for i in range(self.num_of_embeddings)]

        if self.embedding_input_only:
            flat = embeddings  # (N, embed_dim)
        else:
            flat = torch.cat(
                [embeddings, As.view(self.num_of_embeddings, -1)], dim=1
            )  # (N, embed_dim + in_dim*r)

        tokens = self.token_proj(flat).unsqueeze(0)  # (1, N, hidden)

        h = self.transformer(tokens).squeeze(0)  # (N, hidden)

        out = self.out_proj(h).view(self.num_of_embeddings, self.lora_r, self.input_dim)

        if self.hypernet_A_matrix == "generated":
            self.A_matrices = [out[i].view(self.input_dim, self.lora_r) for i in range(self.num_of_embeddings//2)]
            self.B_matrices = [out[i] for i in range(self.num_of_embeddings//2, self.num_of_embeddings)]
        else:
            self.A_matrices = [As[i].detach().clone() for i in range(self.num_of_embeddings)]
            self.B_matrices = [out[i] for i in range(self.num_of_embeddings)]

    def use_precomputed(self, layer_id):
        if self.A_matrices is None or self.B_matrices is None:
            raise RuntimeError(
                "No precomputed matrices found. Call precompute() first."
            )

        return self.A_matrices[layer_id], self.B_matrices[layer_id]


