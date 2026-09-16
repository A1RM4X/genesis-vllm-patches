import urllib.request
import json

# Send request and print details
req = urllib.request.Request(
    "http://localhost:8390/v1/chat/completions",
    data=json.dumps({
        "model": "qwen3.8",
        "messages": [{"role": "user", "content": "Hola"}],
        "max_tokens": 10,
        "temperature": 0.0
    }).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

try:
    with urllib.request.urlopen(req) as response:
        res = json.loads(response.read().decode("utf-8"))
        print(json.dumps(res, indent=2))
except Exception as e:
    print("Error:", e)
