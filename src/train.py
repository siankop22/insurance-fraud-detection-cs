import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from pathlib import Path

RAW_DATA = Path("data/raw/claims.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC = ['policy_age_months','claim_amount','prior_claims','risk_score','provider_tenure','time_since_incident_days','weekday']
CATEGORICAL = ['claim_type','region','channel']

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def build_preprocessor():
    ohe_kwargs = dict(handle_unknown='ignore')
    # Prefer new API (>=1.2): sparse_output
    try:
        OneHotEncoder(sparse_output=False, **ohe_kwargs)
        ohe_kwargs['sparse_output'] = False
    except TypeError:
        # Fallback for older scikit-learn (<1.2)
        ohe_kwargs['sparse'] = False

    ohe = OneHotEncoder(**ohe_kwargs)

    return ColumnTransformer([
        ('num', StandardScaler(), NUMERIC),
        ('cat', ohe, CATEGORICAL),
    ])


def train_and_save():
    df = pd.read_csv(RAW_DATA)
    X, y = df.drop(columns=['is_fraud']), df['is_fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    models = {
        'decision_tree': DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=200, min_samples_leaf=10, n_jobs=-1, random_state=42),
        'mlp': MLPClassifier(hidden_layer_sizes=(64,32), activation='relu', max_iter=200, early_stopping=True, random_state=42)
    }
    prep = build_preprocessor()
    for name, model in models.items():
        pipe = Pipeline([('prep', prep), ('clf', model)])
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, MODEL_DIR / f"{name}.joblib")
        print(f"✅ Trained {name}")

if __name__ == "__main__":
    train_and_save()
