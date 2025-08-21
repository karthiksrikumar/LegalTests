import pandas as pd
import requests
import time
from retry import retry

# API Key (replace with your DeepSeek API key)
DEEPSEEK_API_KEY = "YOUR_API_KEY"

# API endpoint (assumed based on DeepSeek's standard API)
API_URL = "https://api.deepseek.com/v3"

# Load CSV
df = pd.read_csv("cases.csv")

# Prompt template
PROMPT_TEMPLATE = """Given the following case description, answer in **one word** whether the case was affirmed or reversed by the court. The answer must be either "Affirmed" or "Reversed".

Case Description: {description}

Answer:
"""

# Function to query DeepSeek
@retry(tries=3, delay=2, backoff=2)
def query_model(description):
    try:
        headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
        payload = {
            "model": "deepseek-v3",
            "messages": [{"role": "user", "content": PROMPT_TEMPLATE.format(description=description)}],
            "max_tokens": 1,
            "temperature": 0.0
        }
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"].strip()
        return answer if answer in ["Affirmed", "Reversed"] else None
    except Exception as e:
        print(f"Error: {e}")
        return None

# Process cases and store results for metrics
results = []
for _, row in df.iterrows():
    case_title = row["Case Title"]
    description = row["Description"]
    true_judgment = row["Affirm/Reverse"]
    predicted = query_model(description)
    print(f"Case: {case_title}, Predicted: {predicted}, True: {true_judgment}")
    results.append({"Case Title": case_title, "True Judgment": true_judgment, "Predicted Judgment": predicted})
    time.sleep(0.5)  # Avoid rate limits

# Save results for metrics (temporary)
pd.DataFrame(results).to_csv("deepseek_temp_results.csv", index=False)
print("DeepSeek V3 processing complete.")
