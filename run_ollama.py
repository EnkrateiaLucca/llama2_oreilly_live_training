import ollama


response = ollama.chat(
    # model=""
    model="llama3.2",
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant \
            that helps users craft and send their emails using the send_email tool.'},
        {'role': 'user', 'content': 'Can you send an email \
            to lucasbnsoares@hotmail.com, containing a nice motivational message?"'},
    ],
    tools=[{
        'type': 'function',
        'name': 'send_email',
        'description': 'Send an email to a user, the input params are to,\
            subject and body.',
        'parameters': {
            'type': 'object',
            'properties': {
                'to': {
                    'type': 'string',
                    'description': 'The email address of the receiver',
                },
                'subject': {
                    'type': 'string',
                    'description': 'The subject of the email',
                },
                'body': {
                    'type': 'string',
                    'description': 'The body of the email',
                }
            },
            'required': ['to', 'subject', 'body']
        }
    }]
)

print(response)