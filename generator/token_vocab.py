'这个文件负责把公式转成 token，也负责把 token 转回公式'


import re


SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]


BASE_TOKENS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "vwap",
    "returns",

    "ts_mean",
    "ts_std",
    "ts_min",
    "ts_max",
    "ts_rank",
    "delta",
    "delay",
    "zscore",
    "signed_log",
    "abs",
    "safe_div",

    "(",
    ")",
    ",",
    "+",
    "-",
    "*",
    "/",

    "1",
    "3",
    "5",
    "10",
    "20",
    "30",
]


class TokenVocab:
    def __init__(self):
        self.tokens = SPECIAL_TOKENS + BASE_TOKENS
        self.token_to_id = {t: i for i, t in enumerate(self.tokens)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

        self.pad_id = self.token_to_id["<PAD>"]
        self.bos_id = self.token_to_id["<BOS>"]
        self.eos_id = self.token_to_id["<EOS>"]
        self.unk_id = self.token_to_id["<UNK>"]

    def __len__(self):
        return len(self.tokens)

    def tokenize(self, expression):
        pattern = r"(safe_div|signed_log|ts_mean|ts_std|ts_min|ts_max|ts_rank|delta|delay|zscore|abs|open|high|low|close|volume|amount|vwap|returns|\d+|[()+\-*/,])"
        tokens = re.findall(pattern, expression)
        return tokens

    def encode(self, expression, max_len=80):
        tokens = ["<BOS>"] + self.tokenize(expression) + ["<EOS>"]

        ids = [self.token_to_id.get(t, self.unk_id) for t in tokens]

        if len(ids) < max_len:
            ids = ids + [self.pad_id] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
            ids[-1] = self.eos_id

        return ids

    def decode(self, ids):
        tokens = []

        for idx in ids:
            token = self.id_to_token.get(int(idx), "<UNK>")

            if token in ["<PAD>", "<BOS>"]:
                continue

            if token == "<EOS>":
                break

            tokens.append(token)

        return self.detokenize(tokens)

    def detokenize(self, tokens):
        expr = ""

        for t in tokens:
            if t == ",":
                expr += ", "
            elif t == "(":
                expr += "("
            elif t == ")":
                expr += ")"
            elif t in ["+", "-", "*", "/"]:
                expr += f" {t} "
            else:
                if len(expr) > 0 and expr[-1].isalnum() and t not in [")"]:
                    expr += " "
                expr += t

        return expr