import torch
from torch import nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(
        self, vocab_size, embed_dim, hidden_size, output_size, num_layers=1, dropout=0.0
    ):
        super().__init__()

        # saves parameters so that they can be saved and loaded later
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Layers
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    # B: Batch size           – number of sequences processed in parallel
    # L: Sequence length      – number of time steps (tokens) in each sequence
    # D: Embedding dimension  – size of each token’s embedding vector
    # H: Hidden size          – number of features in the LSTM hidden state
    # C: Number of classes    – dimensionality of the output logits

    def forward(self, x):
        # x: LongTensor of shape [B, L] or [B, L, 1]
        if x.dim() == 3 and x.size(2) == 1:
            x = x.squeeze(2)  # remove channel dimension → [B, L]

        emb = self.embedding(x)  # embeddings → [B, L, D]

        # LSTM returns:
        # - output: hidden state at each time step → [B, L, H]
        # - hidden: final hidden state for each layer → [num_layers, B, H]
        # not used as we only need the last hidden state, but can be useful for debugging
        output, (hidden, _) = self.lstm(emb)

        # hidden[-1] selects the final hidden state of the top (last) layer
        # at the last time step → [B, H]
        last_hidden = hidden[-1]

        # apply the fully-connected layer to get logits → [B, C]
        logits = self.fc(last_hidden)

        return logits

    @torch.no_grad
    def predict(self, input, batch_size):
        x = torch.tensor(input, dtype=torch.int)

        outputs = []
        for i in range(0, len(x), batch_size):
            batch = x[i : i + batch_size]
            out = self(batch)
            outputs.append(out)
        outputs = torch.cat(outputs, dim=0)

        return F.softmax(outputs, dim=1)
