
"""
Customer Support Agent System

Architecture:

User
  |
  v
Triage Agent
  |
  +--> Order Status Agent --> lookup_order()
  |
  +--> Refund Agent -------> process_refund()
  |
  +--> FAQ Agent ----------> search_faq()

Uses:
- OpenAI Agents SDK
- Google Gemini through OpenAI-compatible API
- Custom function tools
- Input guardrails
- Pydantic structured output
- Agent handoffs
"""

import os
import asyncio
import warnings

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

from agents import (
    Agent,
    Runner,
    function_tool,
    GuardrailFunctionOutput,
    input_guardrail,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is missing.\n"
        "Create a .env file and add:\n"
        "GOOGLE_API_KEY=your_gemini_api_key"
    )


# Gemini OpenAI-compatible client
client = AsyncOpenAI(
    api_key=GOOGLE_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Tell Agents SDK to use our Gemini client
set_default_openai_client(client)

# Gemini OpenAI compatibility currently uses Chat Completions
set_default_openai_api("chat_completions")

# We are not using an OpenAI API key for tracing
set_tracing_disabled(True)


MODEL = "gemini-3.6-flash"


# ============================================================
# 2. SIMULATED ORDER DATABASE
# ============================================================

ORDERS_DB = {
    "ORD-001": {
        "item": "Wireless Headphones",
        "status": "Shipped",
        "eta": "July 15",
    },
    "ORD-002": {
        "item": "Python Programming Book",
        "status": "Delivered",
        "eta": "July 20",
    },
    "ORD-003": {
        "item": "Smartwatch",
        "status": "Processing",
        "eta": "July 25",
    },
}


# ============================================================
# 3. CUSTOM FUNCTION TOOLS
# ============================================================

@function_tool
def lookup_order(order_id: str) -> str:
    """
    Look up the status and details of an order using its order ID.
    """

    order_id = order_id.strip().upper()

    order = ORDERS_DB.get(order_id)

    if not order:
        return (
            f"Order ID {order_id} was not found. "
            "Please check the order ID and try again."
        )

    return (
        f"Order ID: {order_id}\n"
        f"Item: {order['item']}\n"
        f"Status: {order['status']}\n"
        f"Estimated Arrival: {order['eta']}"
    )


@function_tool
def process_refund(order_id: str, reason: str) -> str:
    """
    Process a refund request for an order.
    """

    order_id = order_id.strip().upper()
    reason = reason.strip()

    order = ORDERS_DB.get(order_id)

    if not order:
        return (
            f"Order ID {order_id} was not found. "
            "Cannot process refund."
        )

    if order["status"].lower() == "processing":
        return (
            f"Refund for Order ID {order_id} cannot be processed "
            "because the order has not shipped yet. "
            "The order can be cancelled instead."
        )

    return (
        f"Refund initiated successfully.\n"
        f"Order ID: {order_id}\n"
        f"Item: {order['item']}\n"
        f"Reason: {reason}\n"
        f"Refund amount will be credited within 5-7 business days."
    )


# ============================================================
# 4. FAQ TOOL
# ============================================================

FAQ_DATA = {
    "return policy": (
        "Customers can generally return eligible products within "
        "30 days of delivery. Products should normally be unused "
        "and in their original condition."
    ),
    "shipping": (
        "Standard shipping usually takes 3-7 business days. "
        "Delivery times may vary depending on location."
    ),
    "refund": (
        "Approved refunds are generally credited within "
        "5-7 business days after processing."
    ),
    "cancellation": (
        "Orders that have not shipped may be eligible for cancellation."
    ),
    "damaged product": (
        "If your product arrives damaged, contact customer support "
        "with your order ID and a description of the damage."
    ),
}


@function_tool
def search_faq(query: str) -> str:
    """
    Search the local FAQ knowledge base for customer support information.

    This replaces WebSearchTool when using Gemini through the
    OpenAI-compatible Chat Completions API.
    """

    query = query.lower().strip()

    results = []

    for topic, answer in FAQ_DATA.items():
        if topic in query or any(
            word in query for word in topic.split()
        ):
            results.append(
                f"Topic: {topic}\nAnswer: {answer}"
            )

    if results:
        return "\n\n".join(results)

    return (
        "No matching FAQ was found in the knowledge base. "
        "Tell the customer that the question requires additional "
        "information from the support team."
    )


# ============================================================
# 5. INPUT GUARDRAIL
# ============================================================

class SupportCheck(BaseModel):
    is_support_question: bool
    reasoning: str


guardrail_checker = Agent(
    name="Support Topic Checker",
    instructions="""
Determine whether the user's message is a customer support question.

Valid customer-support topics include:
- Order status
- Shipping
- Delivery
- Refunds
- Returns
- Cancellations
- Product questions
- Customer-support FAQs

Invalid topics include:
- Personal advice
- Jokes
- Coding help
- Programming questions
- Poetry
- General unrelated conversations

Return is_support_question=True ONLY when the message is related
to customer support.
""",
    output_type=SupportCheck,
    model=MODEL,
)


@input_guardrail
async def support_only(ctx, agent, input):
    """
    Block messages that are unrelated to customer support.
    """

    result = await Runner.run(
        guardrail_checker,
        input,
        context=ctx.context,
    )

    final = result.final_output

    return GuardrailFunctionOutput(
        output_info={
            "reasoning": final.reasoning,
        },
        tripwire_triggered=not final.is_support_question,
    )


# ============================================================
# 6. SPECIALIST AGENTS
# ============================================================

order_agent = Agent(
    name="Order Status Agent",
    handoff_description=(
        "Handles questions about order status, shipping, "
        "and delivery."
    ),
    instructions="""
You are the Order Status Agent.

Help customers check their order status.

Rules:
1. Ask for an Order ID if the customer has not provided one.
2. Use the lookup_order tool to retrieve order information.
3. Never invent order information.
4. Be friendly and professional.
5. Clearly explain the order status and estimated arrival.
""",
    tools=[lookup_order],
    model=MODEL,
)


refund_agent = Agent(
    name="Refund Agent",
    handoff_description=(
        "Handles refund requests, returns, and cancellations."
    ),
    instructions="""
You are the Refund Agent.

Help customers with refunds and returns.

Rules:
1. Ask for the Order ID if it is missing.
2. Ask for the reason if it is missing.
3. Use process_refund only after you have the required information.
4. Never invent order information.
5. Be empathetic and professional.
""",
    tools=[process_refund],
    model=MODEL,
)


faq_agent = Agent(
    name="FAQ Agent",
    handoff_description=(
        "Handles general customer questions, product information, "
        "and frequently asked questions."
    ),
    instructions="""
You are the FAQ Agent.

Help customers with general questions and FAQs.

Use the search_faq tool whenever the customer asks about:
- Return policies
- Shipping
- Refunds
- Cancellations
- Damaged products
- General support information

Do not invent company policies.

If the FAQ database does not contain the answer, clearly say
that the information is not available in the current knowledge base.
Be helpful and concise.
""",
    tools=[search_faq],
    model=MODEL,
)


# ============================================================
# 7. TRIAGE AGENT
# ============================================================

triage_agent = Agent(
    name="Customer Support Triage",
    instructions="""
You are the front-line customer support agent.

Your job is to understand the customer's issue and route them
to the correct specialist.

Routing rules:

- Order status, shipping, or delivery
  -> Order Status Agent

- Refunds, returns, or cancellations
  -> Refund Agent

- General questions, product information, or FAQs
  -> FAQ Agent

Be warm, professional, and concise.

Do not try to solve specialist questions yourself.
Hand off to the appropriate specialist.
""",
    handoffs=[
        order_agent,
        refund_agent,
        faq_agent,
    ],
    input_guardrails=[support_only],
    model=MODEL,
)


# ============================================================
# 8. CUSTOMER REQUEST HANDLER
# ============================================================

async def handle_customer(message: str):
    """
    Process one customer message through the support system.
    """

    print(f"Customer: {message}")

    try:
        result = await Runner.run(
            triage_agent,
            message,
        )

        print(f"Agent: {result.last_agent.name}")
        print(f"Response: {result.final_output}")

    except Exception as e:
        print(f"Request could not be processed.")
        print(f"Error: {type(e).__name__}: {e}")

    print("=" * 70)
    print()


# ============================================================
# 9. DEMO
# ============================================================

async def main():

    print("=" * 70)
    print("       CUSTOMER SUPPORT AGENT SYSTEM")
    print("=" * 70)
    print()

    # Test 1: Order status
    await handle_customer(
        "Where is my order ORD-001?"
    )

    # Test 2: Refund
    await handle_customer(
        "I want a refund for order ORD-002. "
        "The book arrived damaged."
    )

    # Test 3: FAQ
    await handle_customer(
        "What is the return policy?"
    )

    # Test 4: Off-topic
    await handle_customer(
        "Can you help me write a poem about a cat?"
    )


# ============================================================
# 10. START PROGRAM
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())
