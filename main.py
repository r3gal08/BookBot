# TODO: Run locally on phone????? Mistral 4B?
import cv2
import ollama
import time
from functions.llama_text_extract import overlord_text_extract

# TODO: Store in DB (run in container?
# TODO: Give a user options for what "type" of person they want to respond to.
llm_history_local = [
    # Initial system instruction to the assistant
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

local_ollama_model = 'llama3.2'
image_path = './photos/zoom.png'

# ANSI color codes for yellow and green
yellow = "\033[93m"
green = "\033[92m"
reset = "\033[0m"  # Reset to default terminal color

def ask_overlord_text(user: str, user_message: str):
    llm_history_local.append({"role": user, "content": user_message})
    print(llm_history_local)

    # Query ollama LLM with restful API
    stream = ollama.chat(
        model=local_ollama_model,
        messages=llm_history_local,
        stream=True,
    )

    # Record the start time and re-init chars variable
    chars = 0
    start = time.time()

    # TODO: add chat history..... Containerized DB?
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)  # Print to console
        chars += len(chunk['message']['content'])               # Count characters

    end = time.time()  # Record the end time

    # Print the statements with colored text using ANSI codes
    print(f"\n\n{yellow}{'Time taken:':<12} {end-start:.2f} seconds{reset}")
    print(f"{green}{'Chars:':<12} {chars/(end-start):.2f} /second{reset}")

if __name__ == "__main__":
    extracted_text = overlord_text_extract("./photos/photo_gray_full.jpg")
    print("Extracted_text: ", extracted_text)
    ask_overlord_text("user", extracted_text)
