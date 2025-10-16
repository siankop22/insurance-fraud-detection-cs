import joblib, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from pathlib import Path

MODEL_DIR = Path("models"); DATA = Path("data/raw/claims.csv"); FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(DATA); X, y = df.drop(columns=['is_fraud']), df['is_fraud']

trained = {}
for model_file in MODEL_DIR.glob("*.joblib"):
    model = joblib.load(model_file)
    name = model_file.stem
    proba = model.predict_proba(X)[:,1]
    roc = roc_auc_score(y, proba); pr = average_precision_score(y, proba)
    trained[name] = dict(proba=proba, roc=roc, pr=pr)
    print(f"{name}: ROC={roc:.3f}, PR={pr:.3f}")

plt.figure()
for name, obj in trained.items():
    fpr, tpr, _ = roc_curve(y, obj['proba'])
    plt.plot(fpr, tpr, label=f"{name} (AUC={obj['roc']:.3f})")
plt.plot([0,1],[0,1],'--',color='gray'); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves"); plt.legend(); plt.tight_layout(); plt.savefig(FIG_DIR/'roc_curves.png',dpi=200)

plt.figure()
for name, obj in trained.items():
    prec, rec, _ = precision_recall_curve(y, obj['proba'])
    plt.plot(rec, prec, label=f"{name} (AP={obj['pr']:.3f})")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall Curves")
plt.legend(); plt.tight_layout(); plt.savefig(FIG_DIR/'pr_curves.png',dpi=200)
print("✅ Evaluation complete!")
