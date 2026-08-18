"""
Complete Project_ Customer Support Agent System

This brings together EVERYTHING form the course:
- The Agents SDK
- The Response API
- Custom Function tools
- Input Guardrails
- Hosted tools - WeSearchTool
- Structured output with pydantic

Archtitecure:
    User-> Triage Agent-> Order Status Agent(with lookup order tool)
                        ->Refund agent(with process_refund tool)
                        ->FAQ agent( with wweb search)

"""
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()
import asyncio
from pydantic import BaseModel
from agents import(
    Agent,
    Runner,
    function_tool,
    InputGuardrail,
    GuardrailFunctionOutput,
    input_guardrail,
    WebSearchTool,
)

#Part 1: Define Custom Tools

#Simulated Order for Database
ORDERS_DB={
    "ORD-001": {"item": "Wireless Headphones", "status": "Shipped", "eta":"July 15"},
    "ORD-002": {"item": "Python Programming Book", "status": "Delivered", "eta":"July 20"},
    "ORD-003": {"item": "Smartwatch", "status": "Processing", "eta":"July 25"},
}

@function_tool
def lookup_order(order_id: str) -> str:
    """
    Lookup the status of an order by its ID.
    """
    order = ORDERS_DB.get(order_id.upper())
    if order:
        return f"Order ID: {order_id.upper()}, Item: {order['item']}, Status: {order['status']}, Estimated Arrival: {order['eta']}"
    else:
        return f"Order ID: {order_id.upper()} not found. Please check the order ID and try again.  "


@function_tool
def process_refund(order_id: str, reason: str) -> str:
    """
    Process a refund for an order by its ID with a reason.
    """
    order = ORDERS_DB.get(order_id.upper())
    if not order:
        return f"Order ID: {order_id.upper()} not found. Cannot process refund."
    if order["status"].lower() == "processing":
        return f"Refund for Order ID: {order_id.upper()} cannot be processed - order hasn't shipped yet. It can be canceled instead."
    return(
        f"Refund initiated for Order ID: {order_id.upper()}\n"
       f"Item: {order['item']}\n"
       f"Reason: {reason}\n"
       f"Refund amount will be credited within 5-7 business days."
    )


#Part 2: Define the Guardrail

class SupportCheck(BaseModel):
    is_support_question: bool
    reasoning: str

guardrail_checker=Agent(
    name="Support Topic Checker",
    instructions="""
    Determine if the user's message is a customer support question.
    Valid topics: order status, refunds, returns, product questions, shipping, FAQs.
    Invalid topics: personal advice, jokes, coding help, unrelated conversations.
    Return is_support_question=True ONLY for the customer support topics
    """,
    output_type=SupportCheck,
    model="gemini-3.6-flash",
)

@input_guardrail
async def support_only(ctx,agent,input):
    """Only allow customer support questions."""
    result= await Runner.run(guardrail_checker,input,context=ctx.context)
    final=result.final_output_as(SupportCheck)
    return GuardrailFunctionOutput(
        output_info={"reasoning":final.reasoning},
        tripwire_triggered=not final.is_support_question,

    )


#Part 3: Define specialist Agents
order_agent=Agent(
    name="Order Status Agent",
    handoff_description="Handles questions about oder status, shipping and delivery.",
    instructions="""
    You help customers check their order status.
    Use the lookup_order tool to find the order information.
    IF the customer doesn't provide an Order ID, ask for it.
    Be friendly and professional.
    """,
    tools=[lookup_order],
)

refund_agent= Agent(
    name="Refund Agent",
    handoff_description="Handles refund requests,returns, and cancellation.",
    instructions="""
    You help customers with refund and returns.
    Use the process_refund tool to initiate refunds.
    Always ask for the order ID and reason before processing.
    Be empathetic and helpful.
    """,
    tools= [process_refund],
)

faq_agent=Agent(
    name="FAQ_Agent",
    handoff_description="Handles general product questions and frequesntly asked questions.",
    instructions="""
    You answer general customer questions and FAQs.
    Use web search when you need current information.
    Common topics: shipping policies, return window, prodcut details.
    Be helpful and concise.
    """,
    tools= [WebSearchTool()],
)

#Part 4: Triage Agent

triage_agent= Agent(
    name="Customer_Support_Triage",
    instructions="""
    You are a front-line customer support agent.
    Your job is to understand the customer's issue and route them to the right specialist:
    -Order status, shipping, deliver questions->Order Status Agent
    -Refund Requests, returns, cancellations -> Refund Agent
    -General questions, product info,FAQs ->FAQ Agent

    Be warm, professinal, and route qucikly.

    """,
    handoffs=[order_agent,refund_agent,faq_agent],
    input_guardrails=[support_only], #block off topic questions!
)

#Part 5: Run the system
async def handle_customer(message: str):
    """Process customer message through support system."""
    print(f"Customer: {message}")
    try:
        result=await Runner.run(triage_agent,message)
        print(f"{result.last_agent.name}: {result.final_output}")
    except Exception as e:
        print(f"Blocked: This doesn't appear to be a support question.")

    print("="*70)
    print()

async def main():
    print("="*70)
    print("  CUSTOMER SUPPORT SYSTEM -- DEMO")
    print("="*70)
    print()

    #Test 1: Oder status(->Order status agent -> lookup_oder tool)
    await handle_customer("Wher is my ordere ORD-001?")

    #Test 2: Refund Request(->Refund agent-> Process_refund tool)
    await handle_customer("I want a refund for order ORD-002. The book arrrived damaged.")

    #Test 3: General FAQ(->FAQ Agent->Web search tool)
    await handle_customer("What is Amazon return policy?")

    #Test 4: Off-topic(->BLOCKED by guardrail)
    await handle_customer("Can you help me write a poem about cat?")

if __name__ == "__main__":
    asyncio.run(main())