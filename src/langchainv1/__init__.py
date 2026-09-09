import requests
from langchain_ollama import ChatOllama
from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic math expression, e.g. '1234 * 5678'."""
    return str(eval(expression))  # fine for a toy; never eval untrusted input in prod


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    # 1. Turn the city name into coordinates
    geo = requests.get(
        
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
    ).json()
    print("DEBUG geo:", geo)

    if not geo.get("results"):
        return f"Couldn't find a city called {city}."

    loc = geo["results"][0]
    lat, lon = loc["latitude"], loc["longitude"]

    # 2. Get the current weather at those coordinates
    weather = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": lat, "longitude": lon, "current": "temperature_2m"},
    ).json()

    print("DEBUG weather:", weather)
    temp = weather["current"]["temperature_2m"]
    return f"It's currently {temp}°C in {loc['name']}."

def main() -> None:
    llm = ChatOllama(model="llama3.1")
    llm_with_tools = llm.bind_tools([calculator, get_weather])
    tools_by_name = {"calculator": calculator, "get_weather": get_weather}

    print("Ask me anything (weather or math). Type 'quit' to exit.\n")

    while True:
        # 1. Get input from you
        user_input = input("You: ")
        if user_input.lower() in ("quit", "exit"):
            break

        # 2. Start this question's conversation
        messages = [
                    (
                        "system",
                        "You are a helpful assistant with access to exactly two tools: "
                        "a calculator and a weather lookup. "
                        "Rules: "
                        "(1) Only report information that a tool actually returns. "
                        "Never invent weather details like 'sunny', 'high', or 'low' — "
                        "report only the exact temperature the weather tool gives you. "
                        "(2) These are your ONLY tools. If a question needs something else "
                        "(like population), do not invent a tool — just say you can't do that. "
                        "(3) Call each tool only once per question, and trust its result.",
                    ),
                    ("user", user_input),
                ]

        # 3. The agent loop for THIS question
        while True:
            ai_msg = llm_with_tools.invoke(messages)
            messages.append(ai_msg)

            if not ai_msg.tool_calls:
                print("Agent:", ai_msg.content, "\n")
                break

            for call in ai_msg.tool_calls:
                tool_fn = tools_by_name[call["name"]]
                result = tool_fn.invoke(call["args"])
                messages.append(
                    {"role": "tool", "content": result, "tool_call_id": call["id"]}
                )