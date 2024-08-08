import ollama
import requests

def get_current_weather(city):
    """
    Get the current weather for a city.

    Parameters:
    city (str): The name of the city.

    Returns:
    str: The current temperature in the specified city.
    """
    response = f"https://wttr.in/{city}?format=json"
    data = response.json()
    return f"The current temperature in {city} is: {data['current_condition'][0]['temp_C']}°C"

response = ollama.chat(
    model="llama3.1",
    messages=[{'role': 'user', 'content': 'What is the weather in Toronto?'}],
    tools=[{
        'type': 'function',
        'name': 'get_current_weather',
        'description': 'Get the current weather for a city',
        'parameters': {
            'type': 'object',
            'properties': {
                'city': {
                    'type': 'string',
                    'description': 'The name of the city',
                    'required': ['city']
                }
            }
        }
    }]
)

print(response)
