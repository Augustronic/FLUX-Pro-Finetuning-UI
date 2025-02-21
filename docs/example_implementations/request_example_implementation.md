<!-- Creating request with FLUX 1.1 [pro]
To submit an image generation task with FLUX 1.1 [pro] , create a request:
Install requests (e.g. pip install requests) and run -->

import os
import requests

request = requests.post(
    'https://api.us1.bfl.ai/v1/flux-pro-1.1',
    headers={
        'accept': 'application/json',
        'x-key': os.environ.get("BFL_API_KEY"),
        'Content-Type': 'application/json',
    },
    json={
        'prompt': 'A cat on its back legs running like a human is holding a big silver fish with its arms. The cat is running away from the shop owner and has a panicked look on his face. The scene is situated in a crowded market.',
        'width': 1024,
        'height': 768,
    },
).json()
print(request)
request_id = request["id"]

<!-- A successful response will be a json object containing the request's id, that will be used to retrieve the actual result. -->