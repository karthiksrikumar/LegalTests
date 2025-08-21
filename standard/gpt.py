import pandas as pd
import openai
from openai import OpenAI
import time
from retry import retry

# API Key (replace with your OpenAI API key)
OPENAI_API_KEY = "YOUR_API_KEY"

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Load CSV
df = pd.read_csv("cases.csv")

PROMPT_TEMPLATE = """Given the following case description, answer in **one word** whether the case was affirmed or reversed by the court. The answer must be either "Affirmed" or "Reversed".

Case Description: {description}

Answer:
"""

# Function to query GPT-4o
@retry(tries=3, delay=2, backoff=2)
def query_model(description):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a legal expert answering questions about court case judgments."},
                {"role": "user", "content": PROMPT_TEMPLATE.format(description=description)}
            ],
            max_tokens=1,
            temperature=0.0
        )
        answer = response.choices[0].message.content.strip()
        return answer if answer in ["Affirmed", "Reversed"] else None
    except Exception as e:
        print(f"Error: {e}")
        return None

# Process cases and store results for metrics
results = []
for _, row in df.iterrows():
    case_title = row["Case Title"]
    description = row["Description"]  # No shuffling
    true_judgment = row["Affirm/Reverse"]
    predicted = query_model(description)
    print(f"Case: {case_title}, Predicted: {predicted}, True: {true_judgment}")
    results.append({"Case Title": case_title, "True Judgment": true_judgment, "Predicted Judgment": predicted})
    time.sleep(0.5)  # Avoid rate limits

# Save results for metrics (temporary)
pd.DataFrame(results).to_csv("gpt4o_temp_results.csv", index=False)
print("GPT-4o processing complete.")
