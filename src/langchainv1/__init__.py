from langchain_ollama import ChatOllama
from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic math expression, e.g. '1234 * 5678'."""
    return str(eval(expression))  # fine for a toy; never eval untrusted input in prod


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    return f"It's 72°F and sunny in {city}."


def main() -> None:
    llm = ChatOllama(model="llama3.1")
    # Bind BOTH tools now — the model has to pick the right one(s)
    llm_with_tools = llm.bind_tools([calculator, get_weather])

    # A question that needs both tools
    messages = [
        ("user", "What's the weather in Orlando, and what is 15 times 4?")
    ]

    # Map tool names to the actual functions so the loop can dispatch
    tools_by_name = {"calculator": calculator, "get_weather": get_weather}

    while True:
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        # Visibility: see what the model decided this turn
        print(">>> tool_calls:", ai_msg.tool_calls)

        if not ai_msg.tool_calls:
            print("\nFINAL:", ai_msg.content)
            break

        # Could be one OR several tool calls this turn — handle all of them
        for call in ai_msg.tool_calls:
            tool_fn = tools_by_name[call["name"]]
            result = tool_fn.invoke(call["args"])
            messages.append(
                {"role": "tool", "content": result, "tool_call_id": call["id"]}
            )