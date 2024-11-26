from functions.llama_query import ask_overlord
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

image_path = "./photos/photo_gray_full.jpg" # Only works with jpgs for some reason?

if __name__ == "__main__":
    extracted_text = overlord_text_extract(image_path)
    print("Extracted_text:\n", extracted_text)
    print("\nOverlord response:\n")
    ask_overlord("user", extracted_text, llm_history_local)
    print("\n llm_history:\n", llm_history_local)
