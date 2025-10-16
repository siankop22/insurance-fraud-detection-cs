import numpy as np, pandas as pd
from pathlib import Path

def simulate_claims(n=25000, imbalance=0.08, seed=42):
    rng = np.random.default_rng(seed)
    policy_age_months = rng.gamma(10, 6, n)
    claim_amount = rng.lognormal(mean=8.5, sigma=0.6, size=n)
    prior_claims = rng.poisson(0.4, n)
    risk_score = np.clip(rng.normal(0.5, 0.15, n), 0, 1)
    provider_tenure = rng.gamma(5, 8, n)
    time_since_incident_days = rng.integers(0, 120, n)
    weekday = rng.integers(0, 7, n)
    claim_type = rng.choice(['collision','theft','liability','fire'], n, p=[0.5,0.15,0.3,0.05])
    region = rng.choice(['NE','MW','S','W'], n, p=[0.25,0.23,0.32,0.20])
    channel = rng.choice(['agent','online','phone'], n, p=[0.55,0.35,0.10])
    z = 0.8*risk_score + 0.2*(prior_claims>1) + 0.15*(claim_amount>np.percentile(claim_amount,85))
    z += 0.1*(policy_age_months<6) + 0.1*(channel=='online') + 0.05*(claim_type=='theft')
    z -= 0.08*(provider_tenure>np.percentile(provider_tenure,75))
    z = (z-z.mean())/(z.std()+1e-9)
    p = 1/(1+np.exp(-z))
    thresh = np.quantile(p, 1-imbalance)
    y = (p>=thresh).astype(int)
    df = pd.DataFrame(dict(policy_age_months=policy_age_months, claim_amount=claim_amount, prior_claims=prior_claims,
                           risk_score=risk_score, provider_tenure=provider_tenure, time_since_incident_days=time_since_incident_days,
                           weekday=weekday, claim_type=claim_type, region=region, channel=channel, is_fraud=y))
    return df

if __name__ == "__main__":
    out_path = Path("data/raw/claims.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = simulate_claims()
    df.to_csv(out_path, index=False)
    print(f"✅ Simulated dataset saved to {out_path}")
