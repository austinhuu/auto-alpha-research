import numpy as np, pandas as pd
from pathlib import Path
def generate_sample_data(output_path, n_symbols=15, n_days=160, seed=42):
    rng=np.random.default_rng(seed); dates=pd.bdate_range('2023-01-02', periods=n_days); symbols=[f'{i:06d}.SZ' for i in range(1,n_symbols+1)]; rows=[]
    for sym in symbols:
        price=10+rng.normal(0,1); drift=rng.normal(0.0002,0.0005); vol=rng.uniform(0.01,0.03)
        for dt in dates:
            ret=drift+rng.normal(0,vol); op=price*(1+rng.normal(0,vol/3)); cp=price*(1+ret); hp=max(op,cp)*(1+abs(rng.normal(0,vol/2))); lp=min(op,cp)*(1-abs(rng.normal(0,vol/2))); volume=int(rng.integers(500000,5000000)); amount=volume*cp
            rows.append({'datetime':dt.strftime('%Y-%m-%d'),'symbol':sym,'open':round(op,4),'high':round(hp,4),'low':round(lp,4),'close':round(cp,4),'volume':volume,'amount':round(amount,2)})
            price=cp
    df=pd.DataFrame(rows); output_path=Path(output_path); output_path.parent.mkdir(parents=True,exist_ok=True); df.to_csv(output_path,index=False,encoding='utf-8-sig'); return df
if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]; p=root/'sample_data'/'sample_market_data.csv'; df=generate_sample_data(p); print(p, df.shape)
