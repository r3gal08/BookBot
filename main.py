from aiohttp import web
import asyncio
from functions.llama_query import ask_overlord
from functions.llama_text_extract import overlord_text_extract


# TODO: Store in DB (run in container?
# TODO: Give a user options for what "type" of person they want to respond to.
llm_history_local = [
    {
        "role": "system",
        "content": (
            "You are a knowledgeable professor with a deep understanding of literature, "
            "providing additional insight into excerpts from a book. Your explanations should "
            "be clear, engaging, and insightful, focusing on key themes, underlying meanings, "
            "and relevant context. Aim to make complex ideas understandable and memorable."
        ),
    },
]

# Function to handle the /image_upload POST request
async def handle_image(request):
    # Parse the incoming JSON data
    try:
        data = await request.json()
    except Exception as e:
        return web.Response(status=400, text=f"Error parsing JSON: {str(e)}")

    print("data received....")

    # TODO: Re-evaluate this get method
    # Get the base64-encoded image string
    base64_image = data.get("image")
    if not base64_image:
        return web.Response(status=400, text="No image data received.")

    try:
        # Process the image bytes using overlord_text_extract
        extracted_text = overlord_text_extract(base64_image)
        print("Extracted Text:\n", extracted_text)

        # Ask the LLM with the extracted text
        print("\nOverlord response:\n")
        ask_overlord("user", extracted_text, llm_history_local)
        print("\nllm_history_local:\n", llm_history_local)

        # TODO: Not returning to frontend correctly
        return web.Response(status=200, text="Image processed successfully.")
    except Exception as e:
        return web.Response(status=500, text=f"Error processing image: {str(e)}")

# Function to handle the /chat POST request
async def handle_chat(request):
    try:
        # Parse the incoming JSON data
        data = await request.json()
        model = data.get("model")
        prompt = data.get("prompt")

        if not model or not prompt:
            return web.Response(status=400, text="Missing 'model' or 'prompt' in request.")

        # Log the incoming model and prompt
        print(f"Model: {model}\nPrompt: {prompt}")

        # Process the prompt with the LLM and return a response
        response = ask_overlord("user", prompt, llm_history_local)

        # TODO: Not returning to frontend correctly
        return web.json_response({"response": response, "done": True, "context": llm_history_local})

    except Exception as e:
        return web.Response(status=500, text=f"Error processing chat: {str(e)}")


# Setup the aiohttp server
async def init():
    app = web.Application(client_max_size=10 * 1024 * 1024)  # Set max size (10MB here). Will potentially want to tune

    # Define routes for the /chat and /upload endpoints
    app.router.add_post("/chat", handle_chat)  # Chat API endpoint
    app.router.add_post("/image_upload", handle_image)  # File upload endpoint

    return app

if __name__ == "__main__":
    # Run the app on localhost at port 8080
    app = asyncio.run(init())
    # TODO: Will need to un-hardcode this obviously....... (They are local IPs and ports currently)
    web.run_app(app, host="192.168.4.153", port=8080)
