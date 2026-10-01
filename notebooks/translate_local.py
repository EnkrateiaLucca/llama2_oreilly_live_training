# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "ollama>=0.6.3",
# ]
# ///
from ollama import chat
from ollama import ChatResponse
import argparse

MODEL = 'translategemma:4b'

def translate_text(text: str, target_language: str) -> str:
    """
    Translates the given text into the target language using the Gemma4 model.

    Args:
        text (str): The text to be translated.
        target_language (str): The language to translate the text into.

    Returns:
        str: The translated text.
    """
    response: ChatResponse = chat(
        model=MODEL,
        messages=[
            {
                'role': 'user',
                'content': f'''Translate the following text into {target_language}: "{text}". 
                Your output should be ONLY the translation verbatim and nothing else.''',
            },
        ],
    )
    return response.message.content

def parse_text_input(input_str: str) -> str:
    """
    Parses the input string to determine if it's a file path or direct text.

    Args:
        input_str (str): The input string, which can be a file path or direct text.

    Returns:
        str: The content of the file if a valid file path is provided, otherwise returns the input string.
    """
    try:
        with open(input_str, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return input_str

def main():
    parser = argparse.ArgumentParser(description=f'Translate text using the {MODEL} model.')
    parser.add_argument('text', type=str, help='The text to be translated or a file path containing the text.')
    parser.add_argument('--to', type=str, help='The target language to translate into.')
    args = parser.parse_args()
    text_content = parse_text_input(args.text)
    translated_text = translate_text(text_content, args.to)
    print(translated_text)

if __name__ == '__main__':
    main()
    