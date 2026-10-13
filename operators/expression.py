from dataclasses import dataclass
from typing import Optional
@dataclass
class FactorExpression:
    expression: str
    factor_id: Optional[str] = None
    complexity: int = 0
    def __str__(self): return self.expression
