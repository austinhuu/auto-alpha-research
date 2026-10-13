import random
class RandomFactorGenerator:
    def __init__(self, random_seed=42):
        self.rng=random.Random(random_seed)
        self.base_fields=['open','high','low','close','volume','amount','vwap','returns']
        self.windows=[3,5,10,20,30]
    def _base(self): return self.rng.choice(self.base_fields)
    def _window(self): return self.rng.choice(self.windows)
    def _unary_ts(self,x): return f"{self.rng.choice(['ts_mean','ts_std','ts_min','ts_max','ts_rank','delta','delay'])}({x}, {self._window()})"
    def _binary(self,x,y):
        op=self.rng.choice(['add','sub','mul','div'])
        return {'add':f'(({x}) + ({y}))','sub':f'(({x}) - ({y}))','mul':f'(({x}) * ({y}))','div':f'safe_div(({x}), ({y}))'}[op]
    def _transform(self,x):
        op=self.rng.choice(['identity','abs','signed_log','zscore'])
        return x if op=='identity' else f'{op}({x})'
    def generate_one(self):
        t=self.rng.choice([1,2,3,4,5,6])
        if t==1: return self._unary_ts(self._base())
        if t==2: return self._transform(self._unary_ts(self._base()))
        if t==3: return self._binary(self._unary_ts(self._base()), self._unary_ts(self._base()))
        if t==4:
            x=self._base(); w1=self.rng.choice([3,5,10]); w2=self.rng.choice([20,30])
            return f'safe_div(ts_mean({x}, {w1}), ts_mean({x}, {w2})) - 1'
        if t==5:
            w=self._window(); return f'safe_div((close - ts_min(low, {w})), (ts_max(high, {w}) - ts_min(low, {w})))'
        return self._binary(self._unary_ts(self.rng.choice(['close','vwap','returns'])), self._unary_ts(self.rng.choice(['volume','amount'])))
    def generate_many(self,n=100):
        out=[]; seen=set(); trials=0
        while len(out)<n and trials<n*100:
            e=self.generate_one(); trials+=1
            if e not in seen: seen.add(e); out.append(e)
        return out
