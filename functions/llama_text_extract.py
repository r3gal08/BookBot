import base64
import os
from huggingface_hub import InferenceClient
from functions.load_api_token import load_api_token     # Module for loading API token(s)

def overlord_text_extract(base64_image: str) -> str:
    """
    This function takes a Base64-encoded image string, decodes it to bytes,
    processes the image, sends it to a Hugging Face LLM model,
    and returns the extracted text from the image.

    :param base64_image: Base64-encoded image string
    :return: Extracted text from the image
    """

    # Load API token
    try:
        load_api_token()  # Load the API token
    except ValueError as e:
        print(e)
        return None

    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not api_token:
        print("API token not found!")
        return None

    client = InferenceClient(api_key=api_token)

    image_url = f"data:image/jpeg;base64,{base64_image}"

    # Prepare the message payload and send the request to the model
    response_text = ""
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
                            "text": "You are a text extractor. Extract and repeat only the text back to a user.",
                        },
                    ],
                },
            ],
            max_tokens=500,
            stream=True,
    ):
        if "choices" in message and message.choices[0].delta.content:
            response_text += message.choices[0].delta.content

    return response_text.strip()
