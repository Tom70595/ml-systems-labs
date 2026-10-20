import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the two CSVs
cost_df = pd.read_csv("lab03/results/cost_metrics.csv")
train_df = pd.read_csv("lab03/results/training_results.csv")

# Merge them on the 'model' column
df = pd.merge(cost_df, train_df, on="model")

print("Merged Data:")
print(df[["model", "test_accuracy", "params", "latency_b1_ms", "flops_millions"]])

# Plot 1: Accuracy vs. Parameters
plt.figure(figsize=(8, 5))
plt.scatter(df["params"], df["test_accuracy"] * 100, color='blue', s=100)

# Add text labels for each point
for i, row in df.iterrows():
    plt.annotate(row["model"], 
                 (row["params"], row["test_accuracy"] * 100),
                 textcoords="offset points", xytext=(0,10), ha='center')

plt.title("Accuracy vs. Parameters")
plt.xlabel("Number of Parameters")
plt.ylabel("Test Accuracy (%)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("lab03/results/plot_accuracy_vs_params.png", dpi=120)
print("Saved plot_accuracy_vs_params.png")

# Plot 2: Accuracy vs. Latency (Batch 1)
plt.figure(figsize=(8, 5))
plt.scatter(df["latency_b1_ms"], df["test_accuracy"] * 100, color='red', s=100)

for i, row in df.iterrows():
    plt.annotate(row["model"], 
                 (row["latency_b1_ms"], row["test_accuracy"] * 100),
                 textcoords="offset points", xytext=(0,10), ha='center')

plt.title("Accuracy vs. Inference Latency (Batch Size 1)")
plt.xlabel("Latency (ms)")
plt.ylabel("Test Accuracy (%)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("lab03/results/plot_accuracy_vs_latency.png", dpi=120)
print("Saved plot_accuracy_vs_latency.png")