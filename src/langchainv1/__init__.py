from langchain_ollama import ChatOllama
from langchain_core.tools import tool


# 1. Define a tool. The docstring is what the model reads to decide when to use it.
@tool
def calculator(expression: str) -> str:
    """Evaluate a basic math expression, e.g. '1234 * 5678'."""
    return str(eval(expression))  # fine for a toy; never eval untrusted input in prod


def main() -> None:
    # 2. Give the model the tool so it knows the option exists
    llm = ChatOllama(model="llama3.1")
    llm_with_tools = llm.bind_tools([calculator])

    # 3. Start the conversation
    messages = [("user", "What is 1234 * 5678? Use the calculator.")]

    # 4. The agent loop: ask -> maybe call a tool -> feed result back -> repeat
    while True:
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        if not ai_msg.tool_calls:
            # No tool requested -> this is the final answer
            print(ai_msg.content)
            break

        # The model asked to use a tool. Run it and add the result to the conversation.
        for call in ai_msg.tool_calls:
            result = calculator.invoke(call["args"])
            messages.append(
                {"role": "tool", "content": result, "tool_call_id": call["id"]}
            )