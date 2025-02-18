Finetune Details​#Copy link to "Finetune Details"
Query Parameters
finetune_id
string
required
Responses
Expand
200
Successful Response
application/json
Expand
422
Validation Error
application/json
GET
/v1/finetune_details
Selected HTTP client:Python http.client

http.client
Copy content
import http.client

conn = http.client.HTTPSConnection("api.us1.bfl.ai")

headers = { 'X-Key': "YOUR_SECRET_TOKEN" }

conn.request("GET", "/v1/finetune_details", headers=headers)

res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))

Test Request
(GET /v1/finetune_details)
200
422
null
Successful Response

Finetune​#Copy link to "Finetune"
Body
application/json
file_data
string
required
Base64-encoded ZIP file containing training images and, optionally, corresponding captions.

finetune_comment
string
required
Comment or name of the fine-tuned model. This will be added as a field to the finetune_details.

Example
my-first-finetune
trigger_word
string
default: 
TOK
Trigger word for the fine-tuned model.

Example
TOK
mode
string
enum
required
Mode for the fine-tuned model. Allowed values are 'general', 'character', 'style', 'product'. This will affect the caption behaviour. General will describe the image in full detail.

general
character
style
product
iterations
integer
min: 
100
max: 
1000
default: 
300
Number of iterations for fine-tuning.

learning_rate
number | nullable
min: 
0.000001
max: 
0.005
Learning rate for fine-tuning. If not provided, defaults to 1e-5 for full fine-tuning and 1e-4 for lora fine-tuning.

captioning
boolean
default: 
true
Whether to enable captioning during fine-tuning.

priority
string
enum
default: 
quality
Priority of the fine-tuning process. 'speed' will prioritize iteration speed over quality, 'quality' will prioritize quality over speed.

speed
quality
finetune_type
string
enum
default: 
full
Type of fine-tuning. 'lora' is a standard LoRA Adapter, 'full' is a full fine-tuning mode, with a post hoc lora extraction.

lora
full
lora_rank
integer
enum
default: 
32
Rank of the fine-tuned model. 16 or 32. If finetune_type is 'full', this will be the rank of the extracted lora model.

16
32
Responses
Expand
200
Successful Response
application/json
Expand
422
Validation Error
application/json
POST
/v1/finetune
Selected HTTP client:Python http.client

http.client
Copy content
import http.client

conn = http.client.HTTPSConnection("api.us1.bfl.ai")

headers = {
    'Content-Type': "application/json",
    'X-Key': "YOUR_SECRET_TOKEN"
}

conn.request("POST", "/v1/finetune", headers=headers)

res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))

Test Request
(POST /v1/finetune)
200
422
Copy content
{
  "schema": {}
}
Successful Response