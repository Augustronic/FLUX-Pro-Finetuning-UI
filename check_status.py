import http.client
import json

# Your API key
API_KEY = "21006105-1bcc-4969-abab-97e55051d7a3"
FINETUNE_ID = "80a60490-54ea-48e2-b6b3-f2af58bc37f5"

# Setup connection exactly as shown in docs
conn = http.client.HTTPSConnection("api.us1.bfl.ai")

headers = {
    'X-Key': API_KEY,
    'Accept': 'application/json',  # Explicitly request JSON response
    'Content-Type': 'application/json'
}

print(f"Checking status for fine-tune ID: {FINETUNE_ID}")

try:
    # Make request exactly as shown in docs, ensuring we hit the API endpoint
    endpoint = f"/v1/finetune_details?finetune_id={FINETUNE_ID}"
    print(f"\nMaking request to: {conn.host}{endpoint}")
    
    conn.request("GET", endpoint, headers=headers)
    res = conn.getresponse()
    data = res.read()
    
    print(f"\nResponse Status: {res.status}")
    print(f"Response Headers: {dict(res.getheaders())}")
    
    # Try to parse as JSON
    try:
        json_response = json.loads(data.decode('utf-8'))
        print("\nResponse Body (JSON):")
        print(json.dumps(json_response, indent=2))
    except json.JSONDecodeError:
        print("\nResponse Body (raw):")
        print(data.decode('utf-8'))
    
except Exception as e:
    print(f"Error: {e}")
finally:
    conn.close() 