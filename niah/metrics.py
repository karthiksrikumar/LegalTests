import pandas as pd

# List of temporary result files
result_files = [
    "gpt4o_temp_results.csv",
    "gemini_temp_results.csv",
    "llama_temp_results.csv",
    "claude_temp_results.csv",
    "deepseek_temp_results.csv"
]

# Calculate metrics
for file in result_files:
    df = pd.read_csv(file)
    total_cases = len(df)
    correct = (df["True Judgment"] == df["Predicted Judgment"]).sum()
    accuracy = correct / total_cases if total_cases > 0 else 0

    # Recall for Affirmed and Reversed
    affirmed_cases = df[df["True Judgment"] == "Affirmed"]
    reversed_cases = df[df["True Judgment"] == "Reversed"]
    affirmed_correct = affirmed_cases[affirmed_cases["Predicted Judgment"] == "Affirmed"].shape[0]
    reversed_correct = reversed_cases[reversed_cases["Predicted Judgment"] == "Reversed"].shape[0]
    affirmed_total = affirmed_cases.shape[0]
    reversed_total = reversed_cases.shape[0]
    recall_affirmed = affirmed_correct / affirmed_total if affirmed_total > 0 else 0
    recall_reversed = reversed_correct / reversed_total if reversed_total > 0 else 0

    print(f"\nMetrics for {file.replace('_temp_results.csv', '')}:")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total_cases})")
    print(f"Recall (Affirmed): {recall_affirmed:.4f} ({affirmed_correct}/{affirmed_total})")
    print(f"Recall (Reversed): {recall_reversed:.4f} ({reversed_correct}/{reversed_total})")
