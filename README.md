# AI Integration in Decision-Support Systems for Insurance Fraud Detection  
Author: **Thang Sian Kop (@siankop22)**  
Year: **2024**

---
## 🧠 Overview
This repository contains the **research paper** *“AI Integration in Decision-Support Systems for Insurance Fraud Detection”*, authored by Thang Sian Kop as part of an applied academic research project.  

The paper extends prior work on integrating artificial intelligence (AI) into the operations of insurance companies by developing a **decision-support framework** for fraud detection.  
It examines how **Decision Tree**, **Random Forest**, and **Multilayer Perceptron (MLP)** models can improve claim evaluation accuracy while ensuring transparency and ethical governance in automated decision systems.  

The complete **Python implementation, dataset simulation, and model evaluation code** for this paper are available in the **main branch** of this repository.

---

## 📄 Paper Summary
The research demonstrates the feasibility of AI-assisted fraud detection by simulating a realistic insurance claims dataset and evaluating multiple machine learning models.  
It emphasizes:
- **Model interpretability** — ensuring explainable and auditable AI outputs.  
- **Performance evaluation** — using ROC-AUC and PR-AUC metrics for reliability.  
- **Ethical deployment** — aligning automation with human oversight and fairness.  
- **Scalable design** — proposing a hybrid decision-support system integrating data science and claims operations.

---

## 🧪 Results Summary

| Model | ROC-AUC | PR-AUC | Notes |
|--------|----------|--------|-------|
| **Random Forest** | 0.93 | 0.74 | Highest accuracy and generalization |
| **MLP** | 0.91 | 0.71 | Competitive, but computationally intensive |
| **Decision Tree** | 0.86 | 0.61 | Most interpretable, suitable for audit contexts |

The Random Forest model demonstrated the strongest balance between precision and recall, making it suitable for production deployment with interpretability tools (e.g., SHAP or LIME).
- This contains the finalized academic manuscript (PDF format).  
- The **main branch** hosts all experiment scripts, reproducible pipelines, and generated figures.  

---


