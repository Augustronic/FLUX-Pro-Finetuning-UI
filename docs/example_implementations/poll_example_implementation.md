<!-- Poll for result
To retrieve the result, you can poll the get_result endpoint:
This assumes that the request id is stored in a request_id variable and that the os and requests packages were imported as it was done in the previous step. -->

import time

while True:
    time.sleep(0.5)
    result = requests.get(
        'https://api.us1.bfl.ai/v1/get_result',
        headers={
            'accept': 'application/json',
            'x-key': os.environ.get("BFL_API_KEY"),
        },
        params={
            'id': request_id,
        },
    ).json()
    if result["status"] == "Ready":
        print(f"Result: {result['result']['sample']}")
        break
    else:
        print(f"Status: {result['status']}")

<!-- A successful response will be a json object containing the result and the result['sample'] is signed url for retreival. -->