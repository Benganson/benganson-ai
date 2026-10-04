import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

response = client.models.generate_content(
    model="gemini-3.1-flash-image",
    contents="Create a futuristic robot assistant standing in a dark blue and purple laboratory. Cinematic, highly detailed."
)

for part in response.candidates[0].content.parts:
    if part.inline_data:
        image_data = part.inline_data.data

        with open("test_generated_image.png", "wb") as f:
            f.write(image_data)

        print("IMAGE GENERATED SUCCESSFULLY!")
        print("Saved as: test_generated_image.png")