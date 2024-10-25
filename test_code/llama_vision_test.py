import base64
import os
from huggingface_hub import InferenceClient
from functions.load_api_token import load_api_token     # Module for loading API token(s)

try:
    load_api_token()        # Load the API token
except ValueError as e:     # Handle the error (e.g., log it or exit the program)
    print(e)

api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
client = InferenceClient(api_key=api_token)
image_path = '../photos/photo_gray_full.jpg'

with open(image_path, "rb") as f:
    base64_image = base64.b64encode(f.read()).decode("utf-8")
image_url = f"data:image/jpeg;base64,{base64_image}"

for message in client.chat_completion(
        model="meta-llama/Llama-3.2-11B-Vision-Instruct",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                    {
                        "type": "text",
                        "text": "You are a text extractor. Extract and repeat the text back to a user",
                    },
                ],
            },
        ],
        max_tokens=500,
        stream=True,
):
    print(message.choices[0].delta.content, end="")



# for message in client.chat_completion(
#         model="meta-llama/Llama-3.2-11B-Vision-Instruct",
#         messages=[
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image_url", "image_url": {"url": image_url}},
#                     {"type": "text", "text": "Extract the text from this photo"},
#                 ],
#             }
#         ],
#         max_tokens=500,
#         stream=True,
# ):
#     print(message.choices[0].delta.content, end="")