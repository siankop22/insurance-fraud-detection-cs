import joblib
def load_model(name):
    return joblib.load(f"models/{name}.joblib")
