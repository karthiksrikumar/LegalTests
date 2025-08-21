import pandas as pd
import requests
import time
import random
from retry import retry

# API Key (replace with your Hugging Face API key)
HUGGINGFACE_API_KEY = "YOUR_API_KEY"

# API endpoint for Llama 3 8B
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3-8b"

# Load CSV
df = pd.read_csv("cases.csv")

# Shuffling function
def shuffle_triads(text):
    words = text.split()
    triads = [words[i:i+3] for i in range(0, len(words), 3)]
    for triad in triads:
        random.shuffle(triad)
    return " ".join(word for triad in triads for word in triad)

# Prompt template
PROMPT_TEMPLATE = """Given the following case description, answer in **one word** whether the case was affirmed or reversed by the court. The answer must be either "Affirmed" or "Reversed".

Case Description: {description}

Answer:
"""

# Function to query Llama
@retry(tries=3, delay=2, backoff=2)
def query_model(description):
    try:
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        payload = {
            "inputs": PROMPT_TEMPLATE.format(description=description),
            "parameters": {"max_new_tokens": 1, "temperature": 0.0}
        }
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        answer = response.json()[0]["generated_text"].strip().split()[-1]
        return answer if answer in ["Affirmed", "Reversed"] else None
    except Exception as e:
        print(f"Error: {e}")
        return None

# Process cases and store results for metrics
results = []
for _, row in df.iterrows():
    case_title = row["Case Title"]
    description = shuffle_triads(row["Description"])  # Shuffle description
    true_judgment = row["Affirm/Reverse"]
    predicted = query_model(description)
    print(f"Case: {case_title}, Predicted: {predicted}, True: {true_judgment}")
    results.append({"Case Title": case_title, "True Judgment": true_judgment, "Predicted Judgment": predicted})
    time.sleep(0.5)  # Avoid rate limits

# Save results for metrics (temporary)
pd.DataFrame(results).to_csv("llama_temp_results.csv", index=False)
print("Llama 3 8B processing complete.")
