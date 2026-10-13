import numpy as np
import pandas as pd
def select_low_corr_factors(factor_matrix, score_df, top_k=20, max_corr=0.7):
    selected=[]; cand=score_df.dropna(subset=['abs_icir']).sort_values('abs_icir',ascending=False)
    factor_data=factor_matrix[[c for c in factor_matrix.columns if c.startswith('alpha_')]].copy()
    for _,row in cand.iterrows():
        fid=row['factor_id']
        if not selected: selected.append(fid)
        else:
            corrs=[]
            for old in selected:
                pair=factor_data[[fid,old]].replace([np.inf,-np.inf],np.nan).dropna()
                if len(pair)>=20:
                    c=pair[fid].corr(pair[old])
                    if pd.notna(c): corrs.append(abs(c))
            if (max(corrs) if corrs else 0.0) <= max_corr: selected.append(fid)
        if len(selected)>=top_k: break
    out=score_df[score_df['factor_id'].isin(selected)].copy()
    out['selected_rank']=out['factor_id'].map({f:i+1 for i,f in enumerate(selected)})
    return out.sort_values('selected_rank').reset_index(drop=True)
