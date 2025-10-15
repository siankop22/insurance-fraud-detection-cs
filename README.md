# AI Integration in Decision-Support Systems for Insurance Fraud Detection  
**Extending CSU Insurance Company’s AI Implementation Proposal**  
Author: **Thang Sian Kop (@siankop22)**  
Year: **2024**

---

## 🧠 Overview
This project iz incorporating Artificial Intelligence (AI) into insurance operations.  
The project implements and evaluates an **AI-driven decision-support framework** for detecting fraudulent claims using **machine learning models** including:
- Decision Tree (CART)
- Random Forest
- Multilayer Perceptron (MLP)

The focus is on building **interpretable**, **reproducible**, and **ethical** AI systems that assist insurance analysts in making better, faster, and more transparent fraud-related decisions.

---

## 🧩 Key Features
- **Simulated Dataset (25,000 claims)** to preserve privacy while reflecting real-world insurance data distributions  
- **Model Training Pipeline** with `scikit-learn` for preprocessing, scaling, encoding, and evaluation  
- **Performance Metrics** including ROC-AUC, PR-AUC, and confusion matrix visualizations  
- **UML System Architecture Diagram** illustrating the decision-support workflow  
- **Ethical AI Considerations** emphasizing fairness, privacy, and explainability  
- **APA-Formatted Research Paper** (`docs/research_paper.pdf`)

---

## 📊 Methodology Summary

| Step | Description |
|------|--------------|
| **Data Simulation** | Synthetic dataset generated with probabilistic fraud labeling (~8% fraud rate) |
| **Preprocessing** | Standard scaling and one-hot encoding via `ColumnTransformer` |
| **Model Training** | Decision Tree, Random Forest, and MLPClassifier |
| **Evaluation** | ROC-AUC and PR-AUC metrics, along with threshold analysis |
| **Visualization** | ROC and PR plots generated using `matplotlib` |
| **Architecture Design** | UML-inspired system diagram showing end-to-end flow |

---

## 🧪 Results Summary

| Model | ROC-AUC | PR-AUC | Notes |
|--------|----------|--------|-------|
| **Random Forest** | 0.93 | 0.74 | Highest accuracy and generalization |
| **MLP** | 0.91 | 0.71 | Competitive, but computationally intensive |
| **Decision Tree** | 0.86 | 0.61 | Most interpretable, suitable for audit contexts |

The Random Forest model demonstrated the strongest balance between precision and recall, making it suitable for production deployment with interpretability tools (e.g., SHAP or LIME).



