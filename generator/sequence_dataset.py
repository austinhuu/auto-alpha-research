'这个文件把公式表达式做成 PyTorch 训练集。'

import torch
from torch.utils.data import Dataset


class ExpressionSequenceDataset(Dataset):
    def __init__(self, expressions, vocab, max_len=80):
        self.expressions = expressions
        self.vocab = vocab
        self.max_len = max_len

        self.encoded = [
            torch.tensor(self.vocab.encode(expr, max_len=max_len), dtype=torch.long)
            for expr in expressions
        ]

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        seq = self.encoded[idx]

        x = seq[:-1]
        y = seq[1:]

        return x, y