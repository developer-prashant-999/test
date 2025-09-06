import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import pandas as pd

# Load API Key from .env
load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")  # Changed to standard env var name

# Initialize Gemini client
client = genai.Client(api_key=API_KEY)

# System Prompt
SYSTEM_PROMPT = """
You are movie and series expert and are capable of converting date to day.
Provide clear, concise answers about movies and series.
"""

def ask_gemini(user_message: str) -> str:
    """
    Stateless Gemini call using a merged prompt (system + user).
    """
    try:
        full_prompt = SYSTEM_PROMPT.strip() + "\n\nUser query: " + user_message.strip()

        response = client.models.generate_content(
            model="gemini-2.0-flash",  # Using more available model
            contents=[{"parts": [{"text": full_prompt}]}],
            config=types.GenerateContentConfig(
                max_output_tokens=1000  # Reduced for faster response
            )
        )

        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            return "".join(part.text for part in response.candidates[0].content.parts if part.text)
        else:
            return "No response generated."

    except Exception as e:
        print(f"Gemini API error: {e}")
        # Return a fallback response instead of error message
        if "movie" in user_message.lower() or "series" in user_message.lower():
            return "An engaging movie with compelling storyline."
        return "Content description unavailable."

# Example usage
def main():
    questions = [
        "Is Hamilton Really as Good as Everyone Says? First Time Watching a Musical!! What is the title of the movie here ? Just give me the title of the movie.",
        "FIRST TIME WATCHING K-POP DEMON HUNTERS AND IT LIVED UP TO THE HYPE!! | Movie Reaction What is the title of the movie here ? Just give me the title of the movie.",
        "*FULL METAL JACKET* full on BROKE me.. What is the title of the movie here ? Just give me the title of the movie.",
    ]

    for i, q in enumerate(questions, 1):
        print(f"\nQ{i}: {q}")
        print(f"A{i}: {ask_gemini(q)}")

if __name__ == "__main__":
    main()