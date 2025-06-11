import torch
from torch import nn
from pytorch_metric_learning import miners, losses


class PoseBinaryPT(nn.Module):
    def __init__(self, dim, heads, enc_layers, alpha=0.5):
        super().__init__()
        self.dim = dim
        self.alpha = alpha

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads),
            num_layers=enc_layers
        )

        self.embedding = nn.Identity()  # Output of transformer is the embedding
        self.classifier = nn.Linear(dim, 1)  # Binary output

        self.loss_cls = nn.BCEWithLogitsLoss()
        self.loss_triplet = losses.TripletMarginLoss()
        self.miner = miners.MultiSimilarityMiner()

    def forward(self, x):
        """
        x: Tensor of shape [batch_size, seq_len=1, dim]
        Returns:
            logits (for BCE loss), embeddings (for triplet loss)
        """
        x = x.permute(1, 0, 2)  # [seq_len, batch, dim] for Transformer
        x_encoded = self.encoder(x)  # still [seq_len, batch, dim]
        x_encoded = x_encoded.squeeze(0)  # [batch, dim]
        logits = self.classifier(x_encoded).squeeze(1)
        return logits, x_encoded

    def compute_loss(self, x, y):
        logits, embedding = self.forward(x)
        y_float = y.float()

        # Classification Loss
        loss_cls = self.loss_cls(logits, y_float)

        # Triplet Loss
        hard_pairs = self.miner(embedding, y)
        loss_triplet = self.loss_triplet(embedding, y, hard_pairs)

        # Combined
        loss = self.alpha * loss_triplet + (1 - self.alpha) * loss_cls
        return loss, loss_cls.item(), loss_triplet.item()
