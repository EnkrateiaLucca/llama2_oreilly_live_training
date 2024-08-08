from openai import OpenAI

client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)


def check_weather(city: str) -> str:
    """Function that checks the weather."""
    return "It's sunny in " + city
    
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather in San Francisco?"}
]
tools = [
    {
        "type": "function",
        "function": {
            "name": "check_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city, e.g. San Francisco, CA",
                    },
                },
                "required": ["city"],
            },
        }
    },]



response = client.chat.completions.create(model="llama3.1:70b", messages=messages, tools=tools)

print(response)