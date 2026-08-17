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
# Step 3: Configure MCP Client — using a REAL public server
# (DuckDuckGo Search MCP — free, no API key needed)
# --------------------------------------------------

client = MultiServerMCPClient(
    {
        "duckduckgo": {
            "command": r"C:\Projects\oracle_agentic_ai_foundations\venv\Scripts\uvx.exe",
            "args": ["duckduckgo-mcp-server"],
            "transport": "stdio",
        }
    }
)

# --------------------------------------------------
# Step 4: Extract Clean Text from Agent Response
# --------------------------------------------------

def extract_text(response):
    if isinstance(response, list):
        return response[0]["text"]
    return response

# --------------------------------------------------
# Step 5: Conversational Chat Loop
# --------------------------------------------------

async def chat():
    print("=" * 60)
    print("Web Search Agent (DuckDuckGo MCP Server)")
    print("=" * 60)
    print("Ask me anything you'd normally search the web for.")
    print("Type 'exit' or 'quit' to end the conversation.\n")

    tools = await client.get_tools()
    print(f"Loaded tools: {[t.name for t in tools]}\n")

    agent = create_agent(
        model=model,
        tools=tools,
    )

    conversation_history = []

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ("exit", "quit"):
            print("\nAgent: Goodbye!")
            break

        if not user_input:
            continue

        conversation_history.append(("user", user_input))

        result = await agent.ainvoke(
            {
                "messages": conversation_history
            }
        )

        response_text = extract_text(result["messages"][-1].content)
        conversation_history.append(("assistant", response_text))

        print(f"Agent: {response_text}\n")

# --------------------------------------------------
# Step 6: Run
# --------------------------------------------------

if __name__ == "__main__":
    asyncio.run(chat())