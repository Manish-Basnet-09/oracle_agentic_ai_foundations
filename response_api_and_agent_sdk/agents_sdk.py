"""
First agent with agent SDK
The agents SDK makes it easy to build agents.
Just define an Agent with a model, name, and instructions, then run it.
"""

import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from agents import Agent, Runner, set_default_openai_api, set_default_openai_client

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is missing. Add it to your .env file or environment variables.")

# The Agents SDK expects the plain Gemini model name when using Google’s OpenAI-compatible
# endpoint, not a namespaced value such as "models/...".
client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Google’s OpenAI-compatible endpoint works reliably through the Chat Completions API.
set_default_openai_client(client)
set_default_openai_api("chat_completions")

"""
Step 1: Define an Agent
name: The name of the agent
instructions: The instructions for the agent
model: The model to use for the agent
"""
agent = Agent(
    name="History Tutor",
    instructions=(
        "You are a friendly history tutor. You answer history questions clearly and concisely. "
        "Always include an interesting fun fact in your answers."
    ),
    model="gemini-3.6-flash",
)

"""
Step 2: Run the agent
Runner.run_sync() is the synchronous way to execute an agent.
There is also an async version: await Runner.run()
"""
print("---Question 1---")
result = Runner.run_sync(agent, "Who is the first president of Nepal?")
print(result.final_output)
print()

# Run it again with a different question
print("---Question 2---")
result = Runner.run_sync(agent, "What caused World War II?")
print(result.final_output)
print()