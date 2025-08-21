import pandas as pd
import google.generativeai as genai
import time
from retry import retry

# API Key (replace with your Google API key)
GOOGLE_API_KEY = "YOUR_API_KEY"

# Initialize Gemini client
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro")

# Load CSV
df = pd.read_csv("cases

.csv")

# Prompt template
PROMPT_TEMPLATE = """Given the following case description, answer in **one word** whether the case was affirmed or reversed by the court. The answer must be either "Affirmed" or "Reversed".

Case Description: {description}

Answer:
"""

# Function to query Gemini
@retry(tries=3, delay=2, backoff=2)
def query_model(description):
    try:
        response = model.generate_content(
            PROMPT_TEMPLATE.format(description=description),
            generation_config={"max_output_tokens": 1, "temperature": 0.0}
        )
        answer = response.text.strip()
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
pd.DataFrame(results).to_csv("gemini_temp_results.csv", index=False)
print("Gemini 1.5 Pro processing complete.")
