import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMExpressionGenerator(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size=64,
        hidden_size=128,
        num_layers=2,
        dropout=0.1,
        pad_id=0,
    ):
        super().__init__()

        self.pad_id = pad_id

        self.embedding = nn.Embedding(
            vocab_size,
            embed_size,
            padding_idx=pad_id,
        )

        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)

        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
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