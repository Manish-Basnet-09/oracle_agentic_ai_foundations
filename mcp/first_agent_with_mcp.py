import os
import asyncio

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
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
# Step 3: Configure MCP Client
# --------------------------------------------------

client = MultiServerMCPClient(
    {
        "math": {
            "command": "python",
            "args": ["mcp_math_server.py"],  # path to your server file
            "transport": "stdio",
        }
    }
)

# --------------------------------------------------
# Step 4: Run Agent
# --------------------------------------------------

async def run_agent(question: str):
    print(f"\nUser: {question}")
    print("-" * 60)

    # Fetch tools from the MCP server (this launches/talks to the subprocess)
    tools = await client.get_tools()

    agent = create_agent(
        model=model,
        tools=tools,
    )

    result = await agent.ainvoke(
        {
            "messages": [
                ("user", question)
            ]
        }
    )

    response = result["messages"][-1].content

    if isinstance(response, list):
        print("Agent:", response[0]["text"])
    else:
        print("Agent:", response)

# --------------------------------------------------
# Step 5: Test
# --------------------------------------------------

if __name__ == "__main__":

    asyncio.run(
        run_agent(
            "I have a rectangle with width 12 and height 7. "
            "What is its area, and what is the square root of that area?"
        )
    )