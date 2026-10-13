import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from generator.token_vocab import TokenVocab
from generator.sequence_dataset import ExpressionSequenceDataset
from generator.lstm_generator import LSTMExpressionGenerator
from generator.transformer_generator import TransformerExpressionGenerator


class NeuralFactorGenerator:
    def __init__(
        self,
        model_type="lstm",
        max_len=80,
        embed_size=64,
        hidden_size=128,
        num_layers=2,
        dropout=0.1,
        lr=1e-3,
        batch_size=64,
        epochs=30,
        device=None,
    ):
        self.model_type = model_type
        self.max_len = max_len
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.vocab = TokenVocab()

        if model_type == "lstm":
            self.model = LSTMExpressionGenerator(
                vocab_size=len(self.vocab),
                embed_size=embed_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                pad_id=self.vocab.pad_id,
            )

        elif model_type == "transformer":
            self.model = TransformerExpressionGenerator(
                vocab_size=len(self.vocab),
                embed_size=embed_size,
                num_heads=4,
                num_layers=num_layers,
                hidden_size=hidden_size,
                dropout=dropout,
                pad_id=self.vocab.pad_id,
                max_len=max_len,
            )

        else:
            raise ValueError(f"unknown model_type: {model_type}")

        self.model.to(self.device)

    def fit(self, expressions):
        dataset = ExpressionSequenceDataset(
            expressions=expressions,
            vocab=self.vocab,
            max_len=self.max_len,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        criterion = nn.CrossEntropyLoss(
            ignore_index=self.vocab.pad_id,
        )

        self.model.train()

        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            n_batches = 0

            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)

                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                )

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)

            print(f"[Generator Train] epoch={epoch}, loss={avg_loss:.6f}")

    def generate_many(self, n=100, temperature=0.9):
        expressions = []
        seen = set()

        max_trials = n * 20
        trials = 0

        while len(expressions) < n and trials < max_trials:
            expr = self.model.generate(
                vocab=self.vocab,
                max_len=self.max_len,
                temperature=temperature,
                device=self.device,
            )

            trials += 1

            if self._is_basic_valid(expr) and expr not in seen:
                seen.add(expr)
                expressions.append(expr)

        return expressions

    def _is_basic_valid(self, expr):
        import re

        if not expr:
            return False

        expr = expr.strip()

        if len(expr) < 3:
            return False

        # 括号数量必须匹配
        if expr.count("(") != expr.count(")"):
            return False

        valid_keywords = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "vwap",
            "returns",
        ]

        # 必须至少包含一个行情变量
        if not any(k in expr for k in valid_keywords):
            return False

        bad_patterns = [
            "( )",
            ", )",
            "( ,",
            ", ,",
            "+ +",
            "- -",
            "* *",
            "/ /",
            "( +",
            "( -",
            "( *",
            "( /",
            "+ )",
            "- )",
            "* )",
            "/ )",
        ]

        if any(p in expr for p in bad_patterns):
            return False

        # 过滤 10(close) 这种：数字后面直接跟左括号
        if re.search(r"\d+\s*\(", expr):
            return False

        # 过滤 ts_mean(close, 10)(20) 这种：右括号后面直接接左括号
        if re.search(r"\)\s*\(", expr):
            return False

        # 过滤 close(volume) 这种：变量后面直接接左括号
        for var in valid_keywords:
            if re.search(rf"\b{var}\s*\(", expr):
                return False

        # 不能以这些符号开头
        if expr[0] in [",", "+", "*", "/"]:
            return False

        # 不能以这些符号结尾
        if expr[-1] in [",", "+", "-", "*", "/"]:
            return False

        # 尝试编译，提前过滤语法错误
        try:
            compile(expr, "<factor_expr>", "eval")
        except Exception:
            return False

        return True
    
    def sample_with_log_prob(self, temperature=1.0):
        expression, log_prob = self.model.generate_with_log_prob(
            vocab=self.vocab,
            max_len=self.max_len,
            temperature=temperature,
            device=self.device,
        )

        return expression, log_prob
    
    def get_optimizer(self, lr=1e-5):
        return torch.optim.Adam(self.model.parameters(), lr=lr)