import os

os.system("python3 src/data_simulation.py")
os.system("python3 src/train.py")
os.system("python3 src/evaluate.py")

print("✅ All steps completed! Check reports/figures for results.")
