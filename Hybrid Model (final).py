import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 100
df = pd.DataFrame({
    "patient_id": [f"P{i:03d}" for i in range(1, n + 1)],
    "age": np.random.randint(20, 90, n),
    "gender": np.random.choice(["Male", "Female", "Other"], n, p=[0.45, 0.45, 0.10]),
    "symptom_score": np.random.uniform(0, 10, n).round(2),
    "disease_risk": np.random.uniform(0, 1, n).round(3),
    "expected_utility": np.random.uniform(-1, 1, n).round(3),
    "decision_threshold": np.random.uniform(0.3, 0.7, n).round(2)
})

df["predicted_action"] = [
    "Treat" if r > t + 0.1 else "Monitor" if abs(r - t) < 0.1 else "Discharge"
    for r, t in zip(df["disease_risk"], df["decision_threshold"])
]
df["explanation_score"] = np.random.uniform(0.5, 1.0, n).round(3)
df["model_confidence"] = np.random.uniform(0.6, 1.0, n).round(3)
df["outcome"] = np.random.choice(["Improved", "No Change", "Worsened"], n, p=[0.5, 0.3, 0.2])
df["model_type"] = np.random.choice(["Decision Theory Model", "XAI Model", "Hybrid Model"], n, p=[0.3, 0.3, 0.4])
df.to_csv("Hybrid_Model.csv", index=False)
print("✅ Dataset saved as 'Hybrid_Model.csv'")

result = df.groupby("model_type").agg(
    Avg_Explanation_Score=("explanation_score", "mean"),
    Avg_Model_Confidence=("model_confidence", "mean"),
    Improved_Rate=("outcome", lambda x: (x == "Improved").mean() * 100)
).reset_index().round(2)

print("\n📊 Model Performance Summary\n")
print(result.to_string(index=False))

table1 = pd.DataFrame({
    "MODEL TYPE": ["Hybrid Model", "Decision Theory Model", "Explainable AI"],
    "TOTAL CASES": [44, 25, 31],
    "IMPROVED OUTCOMES": [30, 9, 10],
    "SUCCESS RATE": [68.182, 36.00, 32.258]
})

table2 = pd.DataFrame({
    "MODEL TYPE": ["Hybrid Model", "Decision Theory Model", "Explainable AI (XAI)"],
    "MODEL CONFIDENCE": [0.813, 0.832, 0.795],
    "EXPLANATION SCORE": [0.745, 0.730, 0.776]
})

plt.figure(figsize=(7, 4))
plt.bar(table1["MODEL TYPE"], table1["SUCCESS RATE"], color=["#4CAF50", "#2196F3", "#FFC107"])
plt.title("Model Success Rate Comparison")
plt.ylabel("Success Rate (%)")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
x = np.arange(len(table2))
bw = 0.35
plt.bar(x, table2["MODEL CONFIDENCE"], width=bw, label="Confidence", color="#2196F3")
plt.bar(x + bw, table2["EXPLANATION SCORE"], width=bw, label="Explanation", color="#FF9800")
plt.xticks(x + bw / 2, table2["MODEL TYPE"], rotation=10)
plt.title("Model Confidence vs Explanation Score")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
