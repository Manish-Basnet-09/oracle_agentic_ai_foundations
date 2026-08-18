import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError(
        "GOOGLE_API_KEY is missing. Add it to your .env file or environment variables."
    )

# Google’s OpenAI-compatible endpoint expects the model name including the models/ prefix.
# The previous value was retired and caused the 404 response.
client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=api_key,
    timeout=120.0,
    max_retries=2,
)

response = client.chat.completions.create(
    model="models/gemini-3.6-flash",
    messages=[
        {"role": "user", "content": "Explain what an AI agent is in one paragraph."}
    ],
)

print("=" * 60)
print("RESPONSE:")
print("=" * 60)
print(response.choices[0].message.content)
print()

response2 = client.chat.completions.create(
    model="models/gemini-3.6-flash",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful teacher who explains things simply.",
        },
        {"role": "user", "content": "What is the difference between an AI agent and a chatbot?"},
    ],
)

print("=" * 60)
print("RESPONSE WITH INSTRUCTIONS:")
print("=" * 60)
print(response2.choices[0].message.content)
print()