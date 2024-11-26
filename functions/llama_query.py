import ollama
import time

local_ollama_model = 'llama3.2'

# ANSI color codes for yellow and green
yellow = "\033[93m"
green = "\033[92m"
reset = "\033[0m"  # Reset to default terminal color

"""
Sends a user message to the LLM with the given history and returns the LLM's response.

Parameters:
- user (str): The user's name or identifier.
- user_message (str): The message from the user.
- llm_history_local (list): The conversation history, including system instructions.

Returns:
- str: The LLM's response.
"""
def ask_overlord(user: str, user_message: str, llm_history: list):
    llm_history.append({"role": user, "content": user_message})

    # Query ollama LLM with restful API
    stream = ollama.chat(
        model=local_ollama_model,
        messages=llm_history,
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
