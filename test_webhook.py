import urllib.request
import json

url = 'http://localhost:8000/vapi/webhook'
data = {
    'message': {
        'type': 'tool-calls',
        'toolCallList': [
            {
                'id': 'call_123',
                'function': {
                    'name': 'get_institute_info',
                    'arguments': '{"query": "EC curriculum"}'
                }
            }
        ]
    }
}
req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers={'Content-Type': 'application/json'})
try:
    with urllib.request.urlopen(req) as response:
        print(response.read().decode())
except Exception as e:
    print('Error:', e)
