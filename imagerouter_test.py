import requests, json

k = "47e69b92a80eaee50f1e91bc33dade9ea96c8bdffd33df3a30f8362863e625c9"
url = "https://api.imagerouter.io/v1/openai/images/generations"
payload = {
    "prompt": "test image for rate limit check",
    "model": "HiDream-ai/HiDream-I1-Full:free",
    "response_format": "b64_json"
}
headers = {"Authorization": f"Bearer {k}", "Content-Type": "application/json"}

try:
    r = requests.post(url, json=payload, headers=headers, timeout=30)
    print('STATUS', r.status_code)
    try:
        print('HEADERS:', json.dumps(dict(r.headers), indent=2))
    except Exception:
        print('HEADERS raw:', r.headers)
    print('BODY:', r.text)
except Exception as e:
    print('EXC', e)
