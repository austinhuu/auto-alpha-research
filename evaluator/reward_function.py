def calculate_reward(mean_ic, icir, corr_penalty=0.0, complexity_penalty=0.0, lambda_corr=0.5, lambda_complexity=0.01):
    ic_score = abs(icir) if icir is not None else 0.0
    return ic_score - lambda_corr * corr_penalty - lambda_complexity * complexity_penalty
