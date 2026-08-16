import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

response = client.models.generate_content(
    model="models/gemini-3.6-flash",
    contents="What is 43 + 54?"
)

print(response.text)