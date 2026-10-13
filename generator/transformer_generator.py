import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerExpressionGenerator(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size=64,
        num_heads=4,
        num_layers=2,
        hidden_size=128,
        dropout=0.1,
        pad_id=0,
        max_len=80,
    ):
        super().__init__()

        self.pad_id = pad_id
        self.embed_size = embed_size

        self.embedding = nn.Embedding(
            vocab_size,
            embed_size,
            padding_idx=pad_id,
        )

        self.pos_encoder = PositionalEncoding(embed_size, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.fc = nn.Linear(embed_size, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def _causal_mask(self, seq_len, device):
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device),
            diagonal=1,
        ).bool()
        return mask

    def forward(self, x):
        seq_len = x.size(1)
        device = x.device

        padding_mask = x.eq(self.pad_id)

        emb = self.embedding(x) * math.sqrt(self.embed_size)
        emb = self.pos_encoder(emb)

        causal_mask = self._causal_mask(seq_len, device)

        out = self.transformer(
            emb,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )

        logits = self.fc(out)
        return logits

    @torch.no_grad()
    def generate(self, vocab, max_len=80, temperature=1.0, device="cpu"):
        self.eval()

        ids = [vocab.bos_id]

        for _ in range(max_len - 1):
            x = torch.tensor([ids], dtype=torch.long, device=device)
            logits = self.forward(x)

            next_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)

            next_id = torch.multinomial(probs, num_samples=1).item()

            if next_id == vocab.eos_id:
                break

            ids.append(next_id)

        return vocab.decode(ids)
   
    def generate_with_log_prob(
        self,
        vocab,
        max_len=80,
        temperature=1.0,
        device="cpu",
    ):
        self.train()

        ids = [vocab.bos_id]
        log_probs = []

        for _ in range(max_len - 1):
            x = torch.tensor([ids], dtype=torch.long, device=device)

            logits = self.forward(x)
            next_logits = logits[0, -1, :] / temperature

            probs = F.softmax(next_logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)

            next_id = dist.sample()
            log_prob = dist.log_prob(next_id)

            next_id_int = int(next_id.item())

            if next_id_int == vocab.eos_id:
                log_probs.append(log_prob)
                break

            ids.append(next_id_int)
            log_probs.append(log_prob)

        expression = vocab.decode(ids)

        if len(log_probs) == 0:
            total_log_prob = torch.tensor(0.0, device=device)
        else:
            total_log_prob = torch.stack(log_probs).sum()

        return expression, total_log_prob