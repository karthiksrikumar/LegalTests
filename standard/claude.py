import pandas as pd
import anthropic
import time
from retry import retry

# API Key (replace with your Anthropic API key)
ANTHROPIC_API_KEY = "YOUR_API_KEY"

# Initialize Claude client
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Load CSV
df = pd.read_csv("cases.csv")

# Prompt template
PROMPT_TEMPLATE = """Given the following case description, answer in **one word** whether the case was affirmed or reversed by the court. The answer must be either "Affirmed" or "Reversed".

Case Description: {description}

Answer:
"""

# Function to query Claude
@retry(tries=3, delay=2, backoff=2)
def query_model(description):
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1,
            temperature=0.0,
            messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(description=description)}]
        )
        answer = response.content[0].text.strip()
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
pd.DataFrame(results).to_csv("claude_temp_results.csv", index=False)
print("Claude 3.5 Sonnet processing complete.")
