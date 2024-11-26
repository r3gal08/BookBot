import aiohttp
import asyncio
from aiohttp import web
import base64
import io
from PIL import Image

# Function to handle the POST request
async def handle(request):
    # Parse the incoming JSON data
    try:
        # Parse the incoming JSON data
        data = await request.json()
    except Exception as e:
        return web.Response(status=400, text=f"Error parsing JSON: {str(e)}")

    print("data received....")

    # Get the base64-encoded image string
    base64_image = data.get('image')

    if base64_image:
        # Decode the base64 string back to bytes
        image_bytes = base64.b64decode(base64_image)

        # Pass the image bytes to display the image
        await display_image_from_bytes(image_bytes)
        return web.Response(status=200, text="Image received and displayed!")
    else:
        return web.Response(status=400, text="No image data received.")

# Function to display the image
async def display_image_from_bytes(image_bytes: bytes):
    # Create a BytesIO object from the byte data
    image_io = io.BytesIO(image_bytes)

    # Open the image using PIL (Pillow)
    image = Image.open(image_io)

    # Display the image
    image.show()

# Setup the aiohttp server
async def init():
    app = web.Application(client_max_size=10 * 1024 * 1024)  # Set max size (10MB here). Will potentially want to tune
    app.router.add_post('/upload', handle)  # Define the route for the upload
    return app

if __name__ == '__main__':
    # Run the app on localhost at port 8080
    app = asyncio.run(init())
    web.run_app(app, host='192.168.4.153', port=8080)
