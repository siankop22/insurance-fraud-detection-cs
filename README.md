# 🧠 AI Integration in Decision-Support Systems for Insurance Fraud Detection  
Author: **Thang Sian Kop (@siankop22)**  
Year: **2024**

## 📘 Overview
An interpretable, end-to-end **AI fraud-detection system** demonstrating ethical decision-support modeling for insurance workflows.  
This project integrates **machine learning** into an insurance company’s operations to detect fraudulent claims using a simulated dataset.  


## ⚙️ Key Features
- **Synthetic Data Simulation** – 25 000 records emulating realistic insurance claims  
- **End-to-End Pipeline** – preprocessing, encoding, scaling, training, and model persistence  
- **Evaluation Metrics** – ROC-AUC and PR-AUC with visual plots  
- **Reproducible Design** – deterministic random seed for repeatable results  
- **Explainability-Ready** – compatible with SHAP and LIME for future interpretability  

## 🧩 Tech Stack
- **Language:** Python 3.9 +  
- **Libraries:** pandas · numpy · scikit-learn · matplotlib · joblib  
- **Environment:** macOS + VS Code  

## 📂 Repository Structure
```
insurance-fraud-detection-cs/
├── src/
│   ├── data_simulation.py     # Generates synthetic insurance claims dataset
│   ├── train.py               # Trains DecisionTree, RandomForest, and MLP models
│   ├── evaluate.py            # Evaluates models and generates ROC/PR plots
│   └── utils.py               # Helper functions for loading models
├── requirements.txt           # Python dependencies
├── .gitignore                 # Ignored folders/files (data, models, etc.)
├── README.md                  # Project documentation
├── data/                      # Auto-created dataset directory
├── models/                    # Trained models (.joblib)
└── reports/figures/           # ROC and PR curve images
```

## 🚀 How to Run (on macOS or Linux)

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/siankop22/insurance-fraud-detection-cs.git
cd insurance-fraud-detection-cs
```

### 2️⃣ Create and Activate a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Pipeline
```bash
python3 src/data_simulation.py     # Generate dataset
python3 src/train.py               # Train models
python3 src/evaluate.py            # Evaluate & plot results
```

✅ Outputs created automatically:
```
data/raw/claims.csv
models/
reports/figures/roc_curves.png
reports/figures/pr_curves.png
```

## 🧾 Example Results
| Model | ROC-AUC | PR-AUC | Notes |
|:------|:---------|:-------|:------|
| Random Forest | 0.93 | 0.74 | Best balance of precision and recall |
| MLP | 0.91 | 0.71 | Captures non-linear patterns |
| Decision Tree | 0.86 | 0.61 | Most interpretable, audit-friendly |

## 🧮 Research Context
This repository implements the experimental component of *Kop (2023)* — **“Incorporating AI into CSU Insurance Company.”**  
The project explores:
- Interpretability and trust in AI-based decision support  
- Model transparency and ethical AI governance  
- Scalable integration into claims operations  

## 🧭 Future Work
- Integrate SHAP / LIME for explainability  
- Apply cost-sensitive learning for fraud impact optimization  
- Experiment with temporal models (LSTM / Transformers)  
- Deploy interactive UI via Streamlit or Flask  
- Implement continuous monitoring with MLflow or EvidentlyAI  

## 📜 License
Released under the [MIT License](LICENSE).

## ✨ Acknowledgments
Developed by **Thang Sian Kop** .  
Special thanks to mentors and peers supporting human-centered AI and ethical data science.