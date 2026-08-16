import math
import os

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_agent

# --------------------------------------------------
# Step 1: Load Environment Variables
# --------------------------------------------------

load_dotenv()

# --------------------------------------------------
# Step 2: Initialize Gemini Model
# --------------------------------------------------

model = ChatGoogleGenerativeAI(
model="models/gemini-3.6-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# --------------------------------------------------
# Step 3: Define Tools
# --------------------------------------------------

@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """Divide one number by another."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


@tool
def square_root(a: float) -> float:
    """Calculate the square root of a number."""
    if a < 0:
        raise ValueError("Cannot calculate square root of a negative number.")
    return math.sqrt(a)

# --------------------------------------------------
# Step 4: Create Agent
# --------------------------------------------------

tools = [add, multiply, divide, square_root]

agent = create_agent(
    model=model,
    tools=tools,
)

# --------------------------------------------------
# Step 5: Run Agent
# --------------------------------------------------

def run_agent(question: str):
    print(f"\nUser: {question}")
    print("-" * 60)

    result = agent.invoke(
        {
            "messages": [
                ("user", question)
            ]
        }
    )

    # Print only the final response
    response = result["messages"][-1].content

    if isinstance(response, list):
        print("Agent:", response[0]["text"])
    else:
        print("Agent:", response)

# --------------------------------------------------
# Step 6: Test
# --------------------------------------------------

if __name__ == "__main__":

    # run_agent("What is 43 + 54?")

    # run_agent("What is 15 multiplied by 3 and divided by 8?")

    run_agent(
        "I have a rectangle with width 12 and height 7. "
        "What is its area, and what is the square root of that area?"
    )