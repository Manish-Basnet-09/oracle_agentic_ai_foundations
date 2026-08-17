import math

from mcp.server.fastmcp import FastMCP

# --------------------------------------------------
# Step 1: Create MCP Server
# --------------------------------------------------

mcp = FastMCP("MathServer")

# --------------------------------------------------
# Step 2: Define Tools
# --------------------------------------------------

@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide one number by another."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


@mcp.tool()
def square_root(a: float) -> float:
    """Calculate the square root of a number."""
    if a < 0:
        raise ValueError("Cannot calculate square root of a negative number.")
    return math.sqrt(a)

# --------------------------------------------------
# Step 3: Run Server (stdio transport)
# --------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")